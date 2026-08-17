"""
SkillForge AI - Application Entry Point
==============================================

This is the main entry point for SkillForge AI.

Usage:
    # Run the API server
    python run.py serve
    
    # Run document ingestion
    python run.py ingest
    
    # Run interactive CLI demo
    python run.py demo
    
    # Run health check
    python run.py health
    
    # Run tests
    python run.py test

Environment Variables:
    GROQ_API_KEY: Required for LLM operations
    WHATSAPP_CLOUD_API_TOKEN: Required for WhatsApp
    WHATSAPP_PHONE_NUMBER_ID: Required for WhatsApp
    WHATSAPP_VERIFY_TOKEN: For webhook security
    
See .env.example for all configuration options.
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def check_environment() -> bool:
    """Check required environment variables."""
    required_vars = ["GROQ_API_KEY"]
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        print("Please set them in your .env file or environment.")
        return False
    
    return True


def serve(args):
    """Run the API server."""
    import uvicorn
    from src.config import settings
    
    print("=" * 60)
    print("🏥 SkillForge AI - API Server")
    print("=" * 60)
    print(f"Tier Level: {settings.TIER_LEVEL.value}")
    print(f"Debug Mode: {settings.DEBUG_MODE}")
    print(f"Host: {settings.API_HOST}")
    print(f"Port: {settings.API_PORT}")
    print("=" * 60)
    
    if not check_environment():
        return 1
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host or settings.API_HOST,
        port=args.port or settings.API_PORT,
        reload=args.reload or settings.DEBUG_MODE,
        log_level=settings.LOG_LEVEL.lower(),
        workers=args.workers if not args.reload else 1
    )
    
    return 0


def ingest(args):
    """Run document ingestion."""
    from src.data.ingestion import DocumentIngestionPipeline, IngestionConfig
    
    print("=" * 60)
    print("📚 SkillForge AI - Document Ingestion")
    print("=" * 60)
    
    if not check_environment():
        return 1
    
    config = IngestionConfig(
        docs_path=args.docs_path,
        vector_store_path=args.vector_store_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=args.rebuild
    )
    
    pipeline = DocumentIngestionPipeline(config)
    result = pipeline.run()
    
    print("\n" + "=" * 60)
    print("INGESTION RESULTS")
    print("=" * 60)
    print(f"✓ Success: {result.success}")
    print(f"✓ Documents processed: {result.documents_processed}")
    print(f"✓ Chunks created: {result.chunks_created}")
    print(f"✓ Duration: {result.duration_seconds:.1f}s")
    
    if result.errors:
        print(f"\n❌ Errors: {len(result.errors)}")
        for error in result.errors:
            print(f"   - {error}")
    
    print("=" * 60)
    
    return 0 if result.success else 1


async def demo_async():
    """Run interactive demo asynchronously."""
    from src.graph.workflow import create_healthcare_graph
    from src.config import SupportedLanguage
    
    print("=" * 60)
    print("🏥 SkillForge AI - Interactive Demo")
    print("=" * 60)
    print("This demo simulates a patient interview.")
    print("Type 'quit' to exit, 'reset' to start over.")
    print("=" * 60)
    
    # Initialize graph
    print("\nInitializing healthcare AI...")
    graph = create_healthcare_graph()
    
    session_id = f"demo_{__import__('uuid').uuid4().hex[:8]}"
    language = "ur"  # Default to Urdu
    
    # Greeting
    print("\n" + "-" * 40)
    print("AI: السلام علیکم! میں آپ کا AI صحت معاون ہوں۔")
    print("    آج میں آپ کی کیا مدد کر سکتا ہوں؟")
    print("-" * 40)
    print("(Hello! I'm your AI health assistant. How can I help you today?)")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\nThank you for using SkillForge AI. Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                session_id = f"demo_{__import__('uuid').uuid4().hex[:8]}"
                print("\n[Session reset. Starting new conversation...]")
                continue
            
            if user_input.lower() == 'english':
                language = "en"
                print("[Switched to English]")
                continue
            
            if user_input.lower() == 'urdu':
                language = "ur"
                print("[Switched to Urdu]")
                continue
            
            # Process message
            print("\n[Processing...]")
            
            result = await graph.process_message(
                session_id=session_id,
                message=user_input,
                language=language
            )
            
            # Display response
            print("\n" + "-" * 40)
            print(f"AI: {result.get('response', 'No response')}")
            print("-" * 40)
            
            # Show triage level
            triage = result.get('triage_level', 'STANDARD')
            is_emergency = result.get('is_emergency', False)
            
            if is_emergency:
                print(f"\n⚠️  EMERGENCY DETECTED - Triage Level: {triage}")
                print(f"    Action: {result.get('emergency_action', 'Seek immediate care')}")
            else:
                print(f"\n[Triage Level: {triage}]")
            
            # Check if interview is complete
            if result.get('interview_complete'):
                print("\n" + "=" * 60)
                print("INTERVIEW COMPLETE")
                print("=" * 60)
                if result.get('soap_note'):
                    print("\nSOAP Note Generated:")
                    print(result['soap_note'][:500] + "..." if len(result.get('soap_note', '')) > 500 else result['soap_note'])
                print("\n[Type 'reset' to start a new conversation]")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'reset' to start over.\n")


def demo(args):
    """Run interactive demo."""
    if not check_environment():
        return 1
    
    asyncio.run(demo_async())
    return 0


def health(args):
    """Run health check."""
    from src.config import settings
    
    print("=" * 60)
    print("🏥 SkillForge AI - Health Check")
    print("=" * 60)
    
    checks = []
    
    # Check Groq API
    print("\n1. Checking Groq API...")
    if settings.GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
            # Quick test
            client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=5
            )
            print("   ✓ Groq API: Connected")
            checks.append(("Groq API", True))
        except Exception as e:
            print(f"   ✗ Groq API: Error - {e}")
            checks.append(("Groq API", False))
    else:
        print("   ✗ Groq API: No API key configured")
        checks.append(("Groq API", False))
    
    # Check Vector Store
    print("\n2. Checking Vector Store...")
    vector_path = Path(settings.CHROMA_PERSIST_DIR)
    if vector_path.exists():
        try:
            from src.agents.tools.rag_search import get_rag_tool
            rag = get_rag_tool()
            results = rag.search("test query", k=1)
            print(f"   ✓ Vector Store: Available ({len(results)} test results)")
            checks.append(("Vector Store", True))
        except Exception as e:
            print(f"   ✗ Vector Store: Error - {e}")
            checks.append(("Vector Store", False))
    else:
        print("   ⚠ Vector Store: Not initialized (run 'ingest' first)")
        checks.append(("Vector Store", False))
    
    # Check WhatsApp Cloud API
    print("\n3. Checking WhatsApp Cloud API...")
    if settings.WHATSAPP_CLOUD_API_TOKEN and settings.WHATSAPP_PHONE_NUMBER_ID:
        try:
            # Just verify config presence and token format (basic)
            if len(settings.WHATSAPP_CLOUD_API_TOKEN) > 20:
                print("   ✓ WhatsApp Cloud API: Configured")
                checks.append(("WhatsApp API", True))
            else:
                print("   ⚠ WhatsApp Cloud API: Token seems too short")
                checks.append(("WhatsApp API", False))
        except Exception as e:
            print(f"   ✗ WhatsApp Cloud API: Error - {e}")
            checks.append(("WhatsApp API", False))
    else:
        print("   ⚠ WhatsApp Cloud API: Not configured")
        checks.append(("WhatsApp API", None))
    
    # Check Dependencies
    print("\n4. Checking Dependencies...")
    required_packages = [
        "fastapi", "uvicorn", "langchain", "langchain_groq",
        "langchain_huggingface", "chromadb", "pydantic", "groq", "gtts"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"   ✗ Missing packages: {', '.join(missing)}")
        checks.append(("Dependencies", False))
    else:
        print("   ✓ All required packages installed")
        checks.append(("Dependencies", True))
    
    # Summary
    print("\n" + "=" * 60)
    print("HEALTH CHECK SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, status in checks if status is True)
    failed = sum(1 for _, status in checks if status is False)
    skipped = sum(1 for _, status in checks if status is None)
    
    for name, status in checks:
        icon = "✓" if status is True else ("✗" if status is False else "⚠")
        print(f"  {icon} {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


def test(args):
    """Run tests."""
    import subprocess
    
    print("=" * 60)
    print("🧪 SkillForge AI - Running Tests")
    print("=" * 60)
    
    cmd = ["pytest", "src/tests/", "-v"]
    
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html"])
    
    if args.pattern:
        cmd.extend(["-k", args.pattern])
    
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SkillForge AI - Healthcare CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py serve                    # Start API server
    python run.py serve --port 8080        # Start on specific port
    python run.py ingest --rebuild         # Rebuild vector store
    python run.py demo                     # Interactive demo
    python run.py health                   # Health check
    python run.py test --coverage          # Run tests with coverage
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run API server")
    serve_parser.add_argument("--host", type=str, help="Host to bind")
    serve_parser.add_argument("--port", type=int, help="Port to bind")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--docs-path", type=str, default="src/data/docs")
    ingest_parser.add_argument("--vector-store-path", type=str, default="src/data/vector_store")
    ingest_parser.add_argument("--chunk-size", type=int, default=500)
    ingest_parser.add_argument("--chunk-overlap", type=int, default=50)
    ingest_parser.add_argument("--rebuild", action="store_true", help="Rebuild from scratch")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Interactive demo")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Health check")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    test_parser.add_argument("--pattern", "-k", type=str, help="Test pattern filter")
    test_parser.add_argument("--markers", "-m", type=str, help="Test markers filter")
    
    args = parser.parse_args()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Setup logging
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    
    # Route to command
    commands = {
        "serve": serve,
        "ingest": ingest,
        "demo": demo,
        "health": health,
        "test": test
    }
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command in commands:
        return commands[args.command](args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())