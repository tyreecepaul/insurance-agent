import sys
import os
import argparse

# add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Insurance Claims Agent")
    parser.add_argument("--eval",   action="store_true", help="Run evaluation harness")
    parser.add_argument("--ingest", action="store_true", help="Re-index all data")
    parser.add_argument("--only",   choices=["policy", "damage", "claims"],
                        help="(with --ingest) index only one collection")
    parser.add_argument("--variant", help="(with --eval) run one ablation variant")
    parser.add_argument("--family",  help="(with --eval) run one query family")
    args = parser.parse_args()
 
    if args.ingest:
        from ingest import main as ingest_main
        ingest_main(only=args.only)
 
    elif args.eval:
        from eval import run_evaluation
        run_evaluation(
            variant_filter=args.variant,
            family_filter=args.family,
        )
 
    else:
        from agent import run_cli
        run_cli()
 
if __name__ == "__main__":
    main()