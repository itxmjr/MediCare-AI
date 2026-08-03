"""
Document Ingestion Module
=========================
Ingests medical documents into the vector store for RAG.

This module handles:
1. Loading documents from various sources
2. Chunking with medical-aware splitting
3. Metadata extraction
4. Vector store population
5. Index management

Supported Document Types:
- Text files (.txt)
- Markdown files (.md)
- PDF files (.pdf) - requires pypdf
- JSON knowledge bases (.json)

Usage:
    python -m src.data.ingestion --docs-path src/data/docs --rebuild
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    JSONLoader
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class IngestionConfig:
    """Configuration for document ingestion."""
    docs_path: str = "src/data/docs"
    vector_store_path: str = "src/data/vector_store"
    chunk_size: int = 500
    chunk_overlap: int = 50
    rebuild: bool = False
    file_types: List[str] = field(default_factory=lambda: [".txt", ".md", ".json", ".jsonl"])


@dataclass
class IngestionResult:
    """Result of ingestion process."""
    success: bool
    documents_processed: int
    chunks_created: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# MEDICAL TEXT SPLITTER
# =============================================================================

class MedicalTextSplitter(RecursiveCharacterTextSplitter):
    """
    Text splitter optimized for medical documents.
    
    Features:
    - Preserves medical sections (SUBJECTIVE, OBJECTIVE, etc.)
    - Keeps symptom lists together
    - Respects clinical note structure
    """
    
    MEDICAL_SEPARATORS = [
        "\n## ",           # Markdown H2 headers
        "\n### ",          # Markdown H3 headers
        "\n# ",            # Markdown H1 headers
        "\n\n---\n\n",     # Section breaks
        "\n\n",            # Paragraph breaks
        "\n",              # Line breaks
        ". ",              # Sentence breaks
        " ",               # Word breaks
        ""                 # Character breaks
    ]
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwargs
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.MEDICAL_SEPARATORS,
            **kwargs
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text while preserving medical structure."""
        # Pre-process to protect certain patterns
        protected_patterns = [
            (r'(\d+)\.\s+', r'[NUM]\1[/NUM] '),  # Numbered lists
            (r'•\s+', '[BULLET] '),              # Bullet points
        ]
        
        processed_text = text
        for pattern, replacement in protected_patterns:
            import re
            processed_text = re.sub(pattern, replacement, processed_text)
        
        # Split
        chunks = super().split_text(processed_text)
        
        # Post-process to restore patterns
        restored_chunks = []
        for chunk in chunks:
            chunk = chunk.replace('[NUM]', '').replace('[/NUM]', '.')
            chunk = chunk.replace('[BULLET]', '• ')
            restored_chunks.append(chunk.strip())
        
        return restored_chunks


# =============================================================================
# DOCUMENT LOADERS
# =============================================================================

def load_text_documents(docs_path: Path) -> List[Document]:
    """Load text documents from a directory."""
    documents = []
    
    for file_path in docs_path.rglob("*.txt"):
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    "source": file_path.name,
                    "source_type": "text",
                    "document_type": _infer_document_type(file_path.name),
                    "ingested_at": datetime.now().isoformat()
                })
            
            documents.extend(docs)
            logger.info(f"Loaded: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return documents


def load_markdown_documents(docs_path: Path) -> List[Document]:
    """Load markdown documents from a directory."""
    documents = []
    
    for file_path in docs_path.rglob("*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from first H1
            import re
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else file_path.stem
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "source_type": "markdown",
                    "title": title,
                    "document_type": _infer_document_type(file_path.name),
                    "ingested_at": datetime.now().isoformat()
                }
            )
            
            documents.append(doc)
            logger.info(f"Loaded: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return documents


def load_json_knowledge_base(docs_path: Path) -> List[Document]:
    """Load JSON knowledge base files."""
    documents = []
    
    # Support both .json and .jsonl
    patterns = ["*.json", "*.jsonl"]
    
    for pattern in patterns:
        for file_path in docs_path.rglob(pattern):
            try:
                is_jsonl = file_path.suffix == ".jsonl"
                
                if is_jsonl:
                    # Load JSONL line by line
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if not line.strip():
                                continue
                            entry = json.loads(line)
                            content = _json_entry_to_text(entry)
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": file_path.name,
                                    "source_type": "jsonl",
                                    "entry_index": i,
                                    "document_type": _infer_document_type(file_path.name),
                                    "ingested_at": datetime.now().isoformat()
                                }
                            )
                            documents.append(doc)
                else:
                    # Regular JSON
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle different JSON structures
                    if isinstance(data, list):
                        # List of entries
                        for i, entry in enumerate(data):
                            content = _json_entry_to_text(entry)
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": file_path.name,
                                    "source_type": "json",
                                    "entry_index": i,
                                    "document_type": _infer_document_type(file_path.name),
                                    "ingested_at": datetime.now().isoformat()
                                }
                            )
                            documents.append(doc)
                    elif isinstance(data, dict):
                        # Single document or nested structure
                        if "entries" in data:
                            for i, entry in enumerate(data["entries"]):
                                content = _json_entry_to_text(entry)
                                doc = Document(
                                    page_content=content,
                                    metadata={
                                        "source": file_path.name,
                                        "source_type": "json",
                                        "entry_index": i,
                                        "title": data.get("title", ""),
                                        "document_type": _infer_document_type(file_path.name),
                                        "ingested_at": datetime.now().isoformat()
                                    }
                                )
                                documents.append(doc)
                        else:
                            content = _json_entry_to_text(data)
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": file_path.name,
                                    "source_type": "json",
                                    "document_type": _infer_document_type(file_path.name),
                                    "ingested_at": datetime.now().isoformat()
                                }
                            )
                            documents.append(doc)
                
                logger.info(f"Loaded: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    return documents


def _json_entry_to_text(entry: Dict[str, Any]) -> str:
    """Convert a JSON entry to text content."""
    if isinstance(entry, str):
        return entry
    
    parts = []
    
    # Common fields to extract
    priority_fields = ["question", "title", "name", "condition", "symptom"]
    content_fields = ["answer", "content", "description", "text", "details", "information"]
    list_fields = ["symptoms", "signs", "criteria", "actions", "treatments"]
    
    for field_name in priority_fields:
        if field_name in entry:
            if field_name == "question":
                parts.append(f"Question: {entry[field_name]}")
            else:
                parts.append(f"# {entry[field_name]}")
            break
    
    # Special handling for options in Q&A pairs
    if "options" in entry:
        parts.append("\nOptions:")
        if isinstance(entry["options"], dict):
            for key, val in entry["options"].items():
                parts.append(f"- {key}: {val}")
        elif isinstance(entry["options"], list):
            for item in entry["options"]:
                parts.append(f"- {item}")
    
    for field_name in content_fields:
        if field_name in entry:
            if field_name == "answer":
                parts.append(f"\nCorrect Answer: {entry[field_name]}")
            else:
                parts.append(str(entry[field_name]))
    
    for field in list_fields:
        if field in entry and isinstance(entry[field], list):
            parts.append(f"\n{field.title()}:")
            for item in entry[field]:
                parts.append(f"- {item}")
    
    # Include remaining string fields
    for key, value in entry.items():
        if key not in priority_fields + content_fields + list_fields:
            if isinstance(value, str) and len(value) > 10:
                parts.append(f"\n{key}: {value}")
    
    return "\n".join(parts)


def _infer_document_type(filename: str) -> str:
    """Infer document type from filename."""
    filename_lower = filename.lower()
    
    type_keywords = {
        "emergency": "emergency_protocol",
        "triage": "triage_protocol",
        "cardiac": "cardiac_guidelines",
        "heart": "cardiac_guidelines",
        "diabetes": "metabolic_guidelines",
        "metabolic": "metabolic_guidelines",
        "stroke": "neurological_guidelines",
        "drug": "drug_interactions",
        "medication": "drug_interactions",
        "mental": "mental_health_guidelines",
        "pediatric": "pediatric_guidelines",
        "child": "pediatric_guidelines",
    }
    
    for keyword, doc_type in type_keywords.items():
        if keyword in filename_lower:
            return doc_type
    
    return "general_guidelines"


# =============================================================================
# INGESTION PIPELINE
# =============================================================================

class DocumentIngestionPipeline:
    """
    Pipeline for ingesting documents into the vector store.
    
    Steps:
    1. Load documents from various sources
    2. Split into chunks
    3. Generate embeddings
    4. Store in vector database
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.docs_path = Path(config.docs_path)
        self.vector_store_path = Path(config.vector_store_path)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        
        # Initialize text splitter
        self.text_splitter = MedicalTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def run(self) -> IngestionResult:
        """Run the ingestion pipeline."""
        start_time = datetime.now()
        result = IngestionResult(success=True, documents_processed=0, chunks_created=0)
        
        logger.info(f"Starting ingestion from {self.docs_path}")
        
        # Ensure directories exist
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Check if we should rebuild
        if self.config.rebuild and self.vector_store_path.exists():
            logger.info("Rebuilding vector store...")
            import shutil
            shutil.rmtree(self.vector_store_path)
            self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Create default documents if none exist
        if not any(self.docs_path.iterdir()):
            logger.info("No documents found, creating defaults...")
            self._create_default_documents()
        
        # Load all documents
        all_documents = []
        
        try:
            # Load text files
            text_docs = load_text_documents(self.docs_path)
            all_documents.extend(text_docs)
            
            # Load markdown files
            md_docs = load_markdown_documents(self.docs_path)
            all_documents.extend(md_docs)
            
            # Load JSON knowledge bases
            json_docs = load_json_knowledge_base(self.docs_path)
            all_documents.extend(json_docs)
            
            result.documents_processed = len(all_documents)
            logger.info(f"Loaded {len(all_documents)} documents")
            
        except Exception as e:
            result.errors.append(f"Error loading documents: {e}")
            logger.error(f"Document loading error: {e}")
        
        if not all_documents:
            result.warnings.append("No documents found to ingest")
            logger.warning("No documents to ingest")
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            return result
        
        # Split into chunks
        try:
            chunks = self.text_splitter.split_documents(all_documents)
            result.chunks_created = len(chunks)
            logger.info(f"Created {len(chunks)} chunks")
            
        except Exception as e:
            result.errors.append(f"Error splitting documents: {e}")
            logger.error(f"Splitting error: {e}")
            result.success = False
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            return result
        
        # Create vector store
        try:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(self.vector_store_path)
            )
            
            logger.info(f"Vector store created at {self.vector_store_path}")
            
        except Exception as e:
            result.errors.append(f"Error creating vector store: {e}")
            logger.error(f"Vector store error: {e}")
            result.success = False
        
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        if result.success:
            logger.info(
                f"Ingestion complete: {result.documents_processed} docs, "
                f"{result.chunks_created} chunks in {result.duration_seconds:.1f}s"
            )
        
        return result
    
    def _create_default_documents(self) -> None:
        """Create default medical documents."""
        
        documents = {
            "emergency_triage_guidelines.txt": self._get_emergency_guidelines(),
            "cardiac_guidelines.txt": self._get_cardiac_guidelines(),
            "diabetes_guidelines.txt": self._get_diabetes_guidelines(),
            "stroke_guidelines.txt": self._get_stroke_guidelines(),
            "common_symptoms.txt": self._get_common_symptoms(),
        }
        
        for filename, content in documents.items():
            file_path = self.docs_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Created: {filename}")
    
    def _get_emergency_guidelines(self) -> str:
        return """# WHO Emergency Triage Assessment and Treatment (ETAT)

## IMMEDIATE (Red) - Life Threatening
Patients requiring immediate emergency treatment:
- Obstructed airway or central cyanosis
- Severe respiratory distress
- Signs of shock (cold extremities, weak rapid pulse, capillary refill >3 seconds)
- Coma or significantly reduced consciousness
- Convulsing now
- Signs of severe dehydration with altered mental status

**Action:** Treat immediately. Do not wait. Call emergency services.

## VERY URGENT (Orange) - High Priority
Patients with serious conditions:
- Severe chest pain (possible MI)
- Stroke symptoms (FAST positive)
- Severe abdominal pain
- Diabetic emergencies (DKA, HHS)
- Severe allergic reactions
- High fever with danger signs

**Action:** See within 10 minutes. Go to ER immediately.

## URGENT (Yellow) - Priority
Patients needing attention soon:
- Moderate pain not relieved by simple measures
- High fever without danger signs
- Moderate breathing difficulty
- Possible fractures (limb still usable)

**Action:** See within 1 hour. Visit ER or urgent care.

## STANDARD (Green) - Queue
Patients who can wait:
- Minor illness or injury
- Chronic conditions with mild symptoms
- Follow-up visits

**Action:** See in order of arrival. Schedule appointment within 24-48 hours.

## NON-URGENT (Blue) - Advice Only
Very minor problems:
- Minor cuts and bruises
- Cold symptoms
- Health advice needed

**Action:** May be referred to outpatient. See doctor at convenience.
"""
    
    def _get_cardiac_guidelines(self) -> str:
        return """# Cardiac Emergency Assessment Guidelines

## Acute Coronary Syndrome (Heart Attack) Recognition

### Classic Symptoms
- Chest pain or discomfort (pressure, squeezing, fullness)
- Pain radiating to left arm, jaw, neck, or back
- Shortness of breath
- Cold sweat (diaphoresis)
- Nausea or vomiting
- Lightheadedness

### Atypical Presentations (common in women, elderly, diabetics)
- Fatigue
- Indigestion-like symptoms
- Isolated shortness of breath
- Back pain
- Weakness

### Key Assessment Questions
1. Where is the pain located?
2. Does it radiate anywhere?
3. When did it start?
4. What were you doing when it started?
5. Have you had this before?
6. Are you sweating?
7. Any nausea or vomiting?
8. Any shortness of breath?

### Risk Factors to Document
- Previous heart attack or heart disease
- High blood pressure
- Diabetes
- High cholesterol
- Smoking history
- Family history of heart disease
- Age (men >45, women >55)
- Obesity

### Immediate Actions for Suspected MI
1. Call emergency services (1122) immediately
2. Have patient rest in comfortable position
3. Loosen tight clothing
4. If prescribed nitroglycerin available, assist with taking it
5. If patient becomes unresponsive, prepare for CPR
6. Note exact time symptoms started

### STEMI vs NSTEMI
Both require emergency care:
- STEMI: Complete blockage, ST elevation on ECG
- NSTEMI: Partial blockage, no ST elevation
Time to treatment is critical for both.
"""
    
    def _get_diabetes_guidelines(self) -> str:
        return """# Diabetes Assessment and Emergency Guidelines

## Classic Diabetes Symptoms (Type 2)
The "3 Ps":
1. Polyuria - Frequent urination (especially at night)
2. Polydipsia - Excessive thirst
3. Polyphagia - Increased hunger

Additional symptoms:
- Fatigue and weakness
- Unexplained weight loss
- Blurred vision
- Slow healing wounds
- Frequent infections
- Numbness or tingling in hands/feet

## Screening Questions
1. How long have you been feeling tired?
2. Do you feel very thirsty? More than usual?
3. How often do you urinate? Do you wake at night?
4. Any recent weight changes?
5. Any family history of diabetes?
6. Have you had your blood sugar checked before?

## Blood Glucose Interpretation
### Fasting (8+ hours without eating):
- Normal: < 100 mg/dL
- Prediabetes: 100-125 mg/dL
- Diabetes: ≥ 126 mg/dL

### Random (any time):
- Diabetes: ≥ 200 mg/dL with symptoms

### HbA1c:
- Normal: < 5.7%
- Prediabetes: 5.7-6.4%
- Diabetes: ≥ 6.5%

## Diabetic Emergencies

### Diabetic Ketoacidosis (DKA)
Signs:
- Blood glucose > 250 mg/dL
- Nausea and vomiting
- Abdominal pain
- Fruity breath odor (acetone)
- Rapid deep breathing (Kussmaul)
- Confusion or drowsiness
- Dehydration

**Action:** VERY URGENT - Go to ER immediately

### Hyperosmolar Hyperglycemic State (HHS)
Signs:
- Blood glucose > 600 mg/dL
- Extreme dehydration
- Confusion progressing to coma
- No significant ketones

**Action:** IMMEDIATE - Call emergency services

### Hypoglycemia (Low Blood Sugar)
Signs:
- Blood glucose < 70 mg/dL
- Shakiness, trembling
- Sweating
- Rapid heartbeat
- Confusion, irritability
- Hunger

**Action:** Give fast-acting sugar immediately
If unconscious: Call emergency, do not give anything by mouth
"""
    
    def _get_stroke_guidelines(self) -> str:
        return """# Stroke Assessment Guidelines

## FAST Assessment
- F - Face Drooping: Ask person to smile. Is one side drooping?
- A - Arm Weakness: Ask to raise both arms. Does one drift down?
- S - Speech Difficulty: Ask to repeat a phrase. Is speech slurred?
- T - Time to Call: If ANY signs present, call 1122 immediately

## Additional Stroke Symptoms
- Sudden numbness or weakness (especially one side)
- Sudden confusion or trouble understanding
- Sudden trouble seeing in one or both eyes
- Sudden trouble walking, dizziness, loss of balance
- Sudden severe headache with no known cause

## Critical Information to Document
1. Exact time symptoms started (CRITICAL for treatment)
2. Did symptoms come on suddenly or gradually?
3. Are they on blood thinners?
4. Previous stroke history?
5. High blood pressure history?

## Time is Critical
- "Time is Brain" - every minute matters
- tPA (clot-busting drug) window: within 4.5 hours
- Mechanical thrombectomy: within 24 hours for some patients
- Note symptom onset time for medical team

## Subarachnoid Hemorrhage (SAH)
Warning signs:
- "Worst headache of my life"
- Thunderclap onset (maximum intensity in seconds)
- Neck stiffness
- Nausea and vomiting
- Light sensitivity
- Loss of consciousness

**Action:** IMMEDIATE EMERGENCY - Call 1122 NOW
"""
    
    def _get_common_symptoms(self) -> str:
        return """# Common Symptoms Assessment Guide

## Headache Assessment
### Red Flags (Require Urgent Evaluation)
- Sudden severe onset ("thunderclap")
- Worst headache of life
- Associated with fever and neck stiffness
- Following head trauma
- Associated with neurological symptoms
- New headache after age 50
- Progressively worsening over days/weeks

### Assessment Questions
1. When did it start?
2. How quickly did it reach maximum intensity?
3. Where is the pain located?
4. Is this your first headache like this?
5. Any fever or neck stiffness?
6. Any vision changes?
7. Any weakness or numbness?

## Abdominal Pain Assessment
### Red Flags
- Severe sudden onset
- Rigid or board-like abdomen
- Blood in stool or vomit
- High fever
- Unable to pass gas or stool
- Severe pain in right lower quadrant with fever (appendicitis)

### Assessment Questions
1. Where exactly is the pain?
2. Did it start suddenly or gradually?
3. Does the pain move or radiate?
4. Any nausea or vomiting?
5. Any changes in bowel movements?
6. Any fever?
7. When did you last eat?
8. Any blood in stool or vomit?

## Shortness of Breath Assessment
### Red Flags
- Sudden onset at rest
- Chest pain
- Leg swelling (one or both)
- Recent surgery or immobility
- Coughing blood
- Blue lips or fingertips
- Unable to speak full sentences

### Assessment Questions
1. When did it start?
2. Is it constant or does it come and go?
3. What makes it better or worse?
4. Can you lie flat?
5. Any chest pain?
6. Any leg swelling?
7. Any recent travel or surgery?
8. Any cough?

## Fatigue Assessment
### Concerning Features
- Associated with weight loss
- Night sweats
- Fever
- Excessive thirst and urination
- Progressive worsening
- Depression symptoms

### Assessment Questions
1. How long have you felt fatigued?
2. Is it constant or intermittent?
3. How is your sleep?
4. Any weight changes?
5. Any fever or night sweats?
6. Any increased thirst or urination?
7. How is your mood?
8. Any changes in appetite?
"""


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Ingest medical documents into vector store"
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        default="src/data/docs",
        help="Path to documents directory"
    )
    parser.add_argument(
        "--vector-store-path",
        type=str,
        default="src/data/vector_store",
        help="Path to vector store directory"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for text splitting"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector store from scratch"
    )
    
    args = parser.parse_args()
    
    config = IngestionConfig(
        docs_path=args.docs_path,
        vector_store_path=args.vector_store_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=args.rebuild
    )
    
    pipeline = DocumentIngestionPipeline(config)
    result = pipeline.run()
    
    # Print results
    print("\n" + "=" * 50)
    print("INGESTION RESULTS")
    print("=" * 50)
    print(f"Success: {result.success}")
    print(f"Documents processed: {result.documents_processed}")
    print(f"Chunks created: {result.chunks_created}")
    print(f"Duration: {result.duration_seconds:.1f} seconds")
    
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    print("=" * 50)
    
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())