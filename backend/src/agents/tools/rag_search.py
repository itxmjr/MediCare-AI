"""
RAG Search Tool
===============
Retrieval-Augmented Generation for clinical guidelines.

This tool grounds the AI in factual medical information by:
1. Loading and chunking medical documents
2. Creating vector embeddings
3. Retrieving relevant context for queries

Documents indexed:
- WHO clinical guidelines
- Emergency protocols
- Drug interaction databases
- Local health protocols

IMPORTANT: RAG provides context, not diagnosis. The AI uses this
information to ask better questions and identify patterns.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger("rag_search")


# =============================================================================
# RAG SEARCH TOOL CLASS
# =============================================================================

class RAGSearchTool:
    """
    RAG Search Tool for medical guidelines retrieval.
    
    This tool:
    1. Indexes medical documents into a vector store
    2. Retrieves relevant passages for queries
    3. Provides context for clinical decision support
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to avoid re-indexing."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the RAG tool."""
        if RAGSearchTool._initialized:
            return
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        
        self.vector_store: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Document directories
        self.docs_dir = Path("src/data/docs")
        self.persist_dir = Path(settings.CHROMA_PERSIST_DIR)
        
        # Initialize or load vector store
        self._initialize_vector_store()
        
        RAGSearchTool._initialized = True
    
    def _initialize_vector_store(self) -> None:
        """Initialize or load the vector store."""
        try:
            # Try to load existing vector store
            if self.persist_dir.exists():
                logger.info(f"Loading existing vector store from {self.persist_dir}")
                self.vector_store = Chroma(
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings
                )
                # Verify it has documents
                if self.vector_store._collection.count() > 0:
                    logger.info(f"Loaded {self.vector_store._collection.count()} documents")
                    return
            
            # Create new vector store
            logger.info("Creating new vector store...")
            self._index_documents()
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            # Create in-memory store with default documents
            self._create_default_store()
    
    def _index_documents(self) -> None:
        """Index documents from the docs directory."""
        documents = []
        
        # Ensure docs directory exists
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default documents if directory is empty
        if not any(self.docs_dir.iterdir()):
            self._create_default_documents()
        
        # Load all supported files recursively
        extensions = [".txt", ".md", ".json", ".jsonl"]
        
        for ext in extensions:
            for file_path in self.docs_dir.rglob(f"*{ext}"):
                try:
                    # Basic loader for non-text files
                    if ext in [".json", ".jsonl"]:
                        # For JSON, we just want to know it's there, 
                        # but rag_search's internal indexer is basic.
                        # We'll use a simple approach or skip if complex.
                        # Actually, better to just support .txt and .md here 
                        # and rely on ingestion.py for JSON.
                        continue
                    
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    docs = loader.load()
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata["source"] = file_path.name
                        doc.metadata["type"] = self._get_document_type(file_path.name)
                        doc.metadata["path"] = str(file_path.relative_to(self.docs_dir))
                    
                    documents.extend(docs)
                    logger.info(f"Loaded {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        if not documents:
            logger.warning("No documents found, creating defaults")
            self._create_default_store()
            return
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(splits)} chunks")
        
        # Create vector store
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        
        logger.info(f"Indexed {len(splits)} document chunks")
    
    def _get_document_type(self, filename: str) -> str:
        """Determine document type from filename."""
        filename_lower = filename.lower()
        if "emergency" in filename_lower:
            return "emergency_protocol"
        elif "drug" in filename_lower or "interaction" in filename_lower:
            return "drug_interaction"
        elif "diabetes" in filename_lower or "metabolic" in filename_lower:
            return "metabolic_guidelines"
        elif "cardiac" in filename_lower or "heart" in filename_lower:
            return "cardiac_guidelines"
        elif "triage" in filename_lower:
            return "triage_protocol"
        else:
            return "general_guidelines"
    
    def _create_default_documents(self) -> None:
        """Create default medical documents."""
        
        # WHO Emergency Triage Guidelines
        emergency_content = """# WHO Emergency Triage Assessment and Treatment (ETAT)

## IMMEDIATE (Red) - Life Threatening
Patients requiring immediate emergency treatment:
- Obstructed airway
- Severe respiratory distress
- Central cyanosis
- Signs of shock (cold extremities, weak rapid pulse, capillary refill >3 seconds)
- Coma or reduced level of consciousness
- Convulsing now
- Signs of severe dehydration with altered mental status

Action: Treat immediately. Do not wait.

## VERY URGENT (Orange) - High Priority
Patients with serious conditions at risk of deterioration:
- Severe chest pain (possible myocardial infarction)
- Stroke symptoms (FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency)
- Severe abdominal pain
- High fever with danger signs
- Diabetic emergencies (DKA, HHS)
- Severe allergic reactions
- Active psychosis with risk to self or others

Action: See within 10 minutes.

## URGENT (Yellow) - Priority
Patients who need attention soon:
- Moderate pain not relieved by simple measures
- High fever without danger signs
- Moderate breathing difficulty
- Possible fractures (limb still usable)
- Moderate dehydration
- Acute psychiatric symptoms (stable)

Action: See within 1 hour.

## STANDARD (Green) - Queue
Patients who can wait:
- Minor illness or injury
- Chronic conditions with mild symptoms
- Follow-up visits
- Non-urgent referrals

Action: See in order of arrival.

## NON-URGENT (Blue) - Advice Only
Very minor problems:
- Minor cuts and bruises
- Cold symptoms
- Routine medication refills
- Health advice

Action: May be seen at end of queue or referred to outpatient.
"""
        
        # Cardiac Emergency Guidelines
        cardiac_content = """# Cardiac Emergency Assessment Guidelines

## Acute Coronary Syndrome (ACS) / Heart Attack

### Red Flag Symptoms
- Chest pain or discomfort (pressure, squeezing, fullness)
- Pain radiating to arm, jaw, neck, or back
- Shortness of breath
- Cold sweat (diaphoresis)
- Nausea or vomiting
- Lightheadedness or dizziness

### Key Questions to Ask
1. "Where exactly is the pain/discomfort?"
2. "Does the pain go anywhere else?" (radiation)
3. "When did it start?"
4. "What were you doing when it started?"
5. "Have you ever had this before?"
6. "Are you sweating?"
7. "Do you feel short of breath?"
8. "Any nausea or vomiting?"

### Risk Factors
- Previous heart attack or heart disease
- High blood pressure
- Diabetes
- High cholesterol
- Smoking
- Family history of heart disease
- Age (men >45, women >55)
- Obesity

### Immediate Actions
If cardiac emergency suspected:
1. Call emergency services immediately (1122 in Pakistan)
2. Have patient rest in comfortable position
3. Loosen tight clothing
4. If patient has prescribed nitroglycerin, assist with taking it
5. If patient becomes unresponsive, begin CPR if trained

### STEMI vs NSTEMI
STEMI (ST-Elevation MI): Complete blockage, requires immediate intervention
NSTEMI (Non-ST-Elevation MI): Partial blockage, still urgent
Both require emergency care. Do not delay transport.
"""
        
        # Diabetes Assessment Guidelines
        diabetes_content = """# Diabetes Assessment and Emergency Guidelines

## Classic Diabetes Symptoms (Type 2)
The "3 Ps":
1. Polyuria - Frequent urination
2. Polydipsia - Excessive thirst
3. Polyphagia - Increased hunger

Additional symptoms:
- Fatigue and weakness
- Unexplained weight loss
- Blurred vision
- Slow healing wounds
- Frequent infections
- Numbness in hands/feet

## Screening Questions for Suspected Diabetes
1. "How long have you been feeling tired?"
2. "Do you feel very thirsty? More than usual?"
3. "How often do you urinate? Do you wake up at night to urinate?"
4. "Have you noticed any weight change recently?"
5. "Does anyone in your family have diabetes?"
6. "Have you ever had your blood sugar checked?"

## Diabetic Emergencies

### Diabetic Ketoacidosis (DKA)
Signs:
- Blood glucose > 250 mg/dL
- Nausea and vomiting
- Abdominal pain
- Fruity breath odor
- Rapid deep breathing (Kussmaul breathing)
- Confusion or decreased consciousness
- Dehydration

Action: VERY URGENT - Go to ER immediately

### Hyperosmolar Hyperglycemic State (HHS)
Signs:
- Blood glucose > 600 mg/dL
- Extreme dehydration
- Confusion, drowsiness, coma
- No significant ketones

Action: IMMEDIATE - Call emergency services

### Hypoglycemia (Low Blood Sugar)
Signs:
- Blood glucose < 70 mg/dL
- Shakiness, trembling
- Sweating
- Hunger
- Confusion
- Irritability
- Fast heartbeat

Action: Give fast-acting sugar immediately (juice, glucose tablets)
If unconscious: Call emergency, do not give anything by mouth

## Blood Glucose Interpretation
- Fasting (8+ hours): Normal < 100 mg/dL, Prediabetes 100-125, Diabetes ≥ 126
- Random: Diabetes if ≥ 200 mg/dL with symptoms
- HbA1c: Normal < 5.7%, Prediabetes 5.7-6.4%, Diabetes ≥ 6.5%
"""
        
        # Stroke Assessment Guidelines
        stroke_content = """# Stroke Assessment Guidelines

## FAST Assessment
F - Face Drooping: Ask the person to smile. Is one side drooping?
A - Arm Weakness: Ask to raise both arms. Does one arm drift downward?
S - Speech Difficulty: Ask to repeat a simple phrase. Is speech slurred?
T - Time to Call Emergency: If any signs present, call 1122 immediately

## Additional Stroke Symptoms
- Sudden numbness or weakness (face, arm, leg - especially one side)
- Sudden confusion or trouble understanding
- Sudden trouble seeing in one or both eyes
- Sudden trouble walking, dizziness, loss of balance
- Sudden severe headache with no known cause

## Types of Stroke
Ischemic Stroke (87% of cases): Blood clot blocks blood vessel
Hemorrhagic Stroke: Blood vessel bursts in brain
TIA (Mini-stroke): Temporary blockage, symptoms resolve within 24 hours
  - TIA is a warning sign - patient needs evaluation

## Key Questions
1. "When did the symptoms start?" (CRITICAL for treatment decisions)
2. "Did the symptoms come on suddenly or gradually?"
3. "Are you on any blood thinners?"
4. "Have you had a stroke before?"
5. "Do you have high blood pressure?"

## Time is Critical
- "Time is Brain" - Every minute without treatment, 1.9 million neurons die
- tPA (clot-busting drug) must be given within 4.5 hours of symptom onset
- Mechanical thrombectomy within 24 hours for large vessel occlusion

## Subarachnoid Hemorrhage (SAH)
Symptoms:
- "Worst headache of my life"
- Thunderclap onset (maximum intensity within seconds)
- Neck stiffness
- Nausea and vomiting
- Sensitivity to light
- Loss of consciousness

Action: IMMEDIATE EMERGENCY - Call 1122 NOW
"""
        
        # Save documents
        docs = {
            "emergency_triage_guidelines.txt": emergency_content,
            "cardiac_guidelines.txt": cardiac_content,
            "diabetes_guidelines.txt": diabetes_content,
            "stroke_guidelines.txt": stroke_content
        }
        
        for filename, content in docs.items():
            file_path = self.docs_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Created {filename}")
    
    def _create_default_store(self) -> None:
        """Create in-memory store with default documents."""
        default_docs = [
            Document(
                page_content="Emergency triage: IMMEDIATE for life-threatening conditions. "
                            "VERY URGENT for serious conditions. URGENT for priority cases. "
                            "STANDARD for non-urgent. Call 1122 for emergencies in Pakistan.",
                metadata={"source": "default", "type": "emergency_protocol"}
            ),
            Document(
                page_content="Cardiac red flags: Chest pain with radiation to arm/jaw, "
                            "sweating, shortness of breath, nausea. Call emergency immediately.",
                metadata={"source": "default", "type": "cardiac_guidelines"}
            ),
            Document(
                page_content="Diabetes symptoms: Excessive thirst, frequent urination, fatigue, "
                            "unexplained weight loss. Classic triad suggests diabetes screening needed.",
                metadata={"source": "default", "type": "metabolic_guidelines"}
            ),
            Document(
                page_content="Stroke FAST: Face drooping, Arm weakness, Speech difficulty, "
                            "Time to call emergency. Time is critical - note symptom onset time.",
                metadata={"source": "default", "type": "stroke_guidelines"}
            )
        ]
        
        self.vector_store = Chroma.from_documents(
            documents=default_docs,
            embedding=self.embeddings
        )
        logger.info("Created default in-memory vector store")
    
    def search(
        self,
        query: str,
        k: int = None,
        filter_type: Optional[str] = None
    ) -> List[str]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results (default from settings)
            filter_type: Optional document type filter
            
        Returns:
            List of relevant document excerpts
        """
        if k is None:
            k = settings.RAG_TOP_K
        
        if self.vector_store is None:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            # Build filter if specified
            where_filter = None
            if filter_type:
                where_filter = {"type": filter_type}
            
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query,
                k=k,
                filter=where_filter
            )
            
            # Extract content
            excerpts = [doc.page_content for doc in results]
            
            logger.info(f"RAG search for '{query[:50]}...' returned {len(excerpts)} results")
            return excerpts
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return []
    
    def search_with_scores(
        self,
        query: str,
        k: int = None,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of dicts with 'content', 'score', 'metadata'
        """
        if k is None:
            k = settings.RAG_TOP_K
        
        if self.vector_store is None:
            return []
        
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=k
            )
            
            filtered_results = []
            for doc, score in results:
                if score >= score_threshold:
                    filtered_results.append({
                        "content": doc.page_content,
                        "score": score,
                        "metadata": doc.metadata
                    })
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"RAG search with scores error: {e}")
            return []
    
    def add_document(self, content: str, metadata: Dict[str, str]) -> bool:
        """
        Add a new document to the vector store.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            True if successful
        """
        try:
            doc = Document(page_content=content, metadata=metadata)
            splits = self.text_splitter.split_documents([doc])
            
            self.vector_store.add_documents(splits)
            logger.info(f"Added document with {len(splits)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global RAG tool instance
_rag_tool: Optional[RAGSearchTool] = None


def get_rag_tool() -> RAGSearchTool:
    """Get the singleton RAG tool instance."""
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = RAGSearchTool()
    return _rag_tool


def search_guidelines(query: str, k: int = 3) -> List[str]:
    """
    Search clinical guidelines.
    
    Args:
        query: Search query (symptom, condition, etc.)
        k: Number of results
        
    Returns:
        List of relevant guideline excerpts
    """
    tool = get_rag_tool()
    return tool.search(query, k=k)


def search_emergency_protocols(query: str) -> List[str]:
    """
    Search emergency protocols specifically.
    
    Args:
        query: Emergency-related query
        
    Returns:
        List of relevant emergency protocol excerpts
    """
    tool = get_rag_tool()
    return tool.search(query, k=5, filter_type="emergency_protocol")


def search_drug_interactions(medications: List[str]) -> List[str]:
    """
    Search for drug interaction information.
    
    Args:
        medications: List of medication names
        
    Returns:
        List of relevant drug interaction information
    """
    tool = get_rag_tool()
    query = f"drug interactions {' '.join(medications)}"
    return tool.search(query, k=3, filter_type="drug_interaction")


def get_symptom_guidance(symptom: str) -> Dict[str, Any]:
    """
    Get comprehensive guidance for a symptom.
    
    Args:
        symptom: Symptom name
        
    Returns:
        Dict with 'guidelines', 'emergency_signs', 'followup_questions'
    """
    tool = get_rag_tool()
    
    # Search for general guidelines
    general = tool.search(f"{symptom} assessment guidelines", k=2)
    
    # Search for emergency signs
    emergency = tool.search(f"{symptom} emergency red flags danger signs", k=2)
    
    # Search for related questions
    questions = tool.search(f"{symptom} screening questions to ask patient", k=2)
    
    return {
        "guidelines": general,
        "emergency_signs": emergency,
        "followup_questions": questions
    }


# =============================================================================
# DOCUMENT INGESTION SCRIPT
# =============================================================================

def ingest_documents(docs_path: str = "src/data/docs") -> None:
    """
    Ingest documents from a directory into the vector store.
    
    Args:
        docs_path: Path to documents directory
    """
    logger.info(f"Ingesting documents from {docs_path}")
    
    # Reset the singleton
    global _rag_tool
    _rag_tool = None
    RAGSearchTool._initialized = False
    RAGSearchTool._instance = None
    
    # Reinitialize (will re-index documents)
    tool = get_rag_tool()
    
    logger.info("Document ingestion complete")


if __name__ == "__main__":
    # Test the RAG search
    logging.basicConfig(level=logging.INFO)
    
    print("Testing RAG Search Tool...")
    
    # Test basic search
    results = search_guidelines("chest pain cardiac emergency")
    print(f"\nSearch for 'chest pain cardiac emergency':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result[:200]}...")
    
    # Test emergency protocols
    results = search_emergency_protocols("stroke symptoms")
    print(f"\nEmergency protocols for 'stroke symptoms':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result[:200]}...")
    
    # Test symptom guidance
    guidance = get_symptom_guidance("fatigue")
    print(f"\nGuidance for 'fatigue':")
    print(f"Guidelines: {len(guidance['guidelines'])} results")
    print(f"Emergency signs: {len(guidance['emergency_signs'])} results")
    print(f"Follow-up questions: {len(guidance['followup_questions'])} results")