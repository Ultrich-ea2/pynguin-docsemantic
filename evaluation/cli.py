# evaluation/cli.py
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from .config import ExperimentConfig, PynguinRunConfig
from .evaluation import PynguinEvaluation

def main():
    parser = argparse.ArgumentParser(description="Scientific evaluation of Pynguin with/without semantics")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-search-time", type=int, default=300)
    parser.add_argument("--max-memory", type=int, default=5000)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--project-path", type=str, default="tmp/evaluation/examples")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    try:
        import pynguin
        print(f"Found pynguin version: {pynguin.__version__}")
    except ImportError:
        print("Error: pynguin module not found. Make sure it's installed.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_BASE_PATH = f"evaluation/pynguin_evaluations/pynguin_evaluation_{timestamp}"
    RUNS_PATH = os.path.join(OUTPUT_BASE_PATH, "runs")
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    os.makedirs(RUNS_PATH, exist_ok=True)

    project_dir = Path(args.project_path)
    modules = [
        os.path.splitext(f.name)[0]
        for f in project_dir.glob("*.py")
        if f.is_file()
    ]
    print(f"Found {len(modules)} modules: {modules}")

    exp_config = ExperimentConfig(
        project_path=args.project_path,
        output_base_path=OUTPUT_BASE_PATH,
        output_runs_path=RUNS_PATH,
        modules=modules,
        num_runs=args.runs,
        max_iterations=args.iterations,
        max_search_time=args.max_search_time,
        max_memory=args.max_memory,
        population_size=args.population_size,
        verbose=args.verbose
    )

    PynguinEvaluation(exp_config, PynguinRunConfig).evaluate()

if __name__ == "__main__":
    main()
