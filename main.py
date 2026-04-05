import sys
import os
import uvicorn
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

from src.ingest import main as ingest_main
from src.agent import run_cli

from src.api import app

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Insurance Claims Agent")
    parser.add_argument("--eval",   action="store_true", help="Run evaluation harness")
    parser.add_argument("--ingest", action="store_true", help="Re-index all data")
    parser.add_argument("--only",   choices=["policy", "damage", "claims"],
                        help="(with --ingest) index only one collection")
    parser.add_argument("--variant", help="(with --eval) run one ablation variant")
    parser.add_argument("--family",  help="(with --eval) run one query family")
    parser.add_argument("--api",    action="store_true", help="Start REST API server")
    args = parser.parse_args()
 
    if args.ingest:
        
        ingest_main(only=args.only)
 
    elif args.eval:
        from eval import run_evaluation
        run_evaluation(
            variant_filter=args.variant,
            family_filter=args.family,
        )
 
    elif args.api:
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", 8000))
        
        print(f"\n Starting Insurance Claims Agent API")
        print(f"   Host: {host}:{port}")
        print(f"   Docs: http://localhost:{port}/docs")
        print(f"   ReDoc: http://localhost:{port}/redoc\n")
        
        uvicorn.run(app, host=host, port=port, log_level="info")
    else:
        run_cli()
 
if __name__ == "__main__":
    main()