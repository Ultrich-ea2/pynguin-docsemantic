import subprocess
import os
import pandas as pd
import time
from datetime import datetime
import argparse
from pathlib import Path
import sys

def run_pynguin(project_path, output_path, module_name, use_semantics=False, iterations=100, verbose=False):
    """Run Pynguin with the specified parameters."""
    # Create a specific report directory within the output path
    report_dir = os.path.join(output_path, "pynguin-report")
    os.makedirs(report_dir, exist_ok=True)

    # Set environment variable
    os.environ["PYNGUIN_DANGER_AWARE"] = "1"

    # Import pynguin directly
    from pynguin import set_configuration, run_pynguin as execute_pynguin
    from pynguin.configuration import (
        Configuration,
        TestCaseOutputConfiguration,
        StatisticsOutputConfiguration,
        StoppingConfiguration,
        SearchAlgorithmConfiguration,
        SeedingConfiguration,
        Algorithm
    )
    import dataclasses

    # Create configuration with proper nested configuration objects
    config = Configuration(
        project_path=project_path,
        module_name=module_name,
        test_case_output=TestCaseOutputConfiguration(
            output_path=output_path,
            assertion_generation="MUTATION_ANALYSIS"
        ),
        statistics_output=StatisticsOutputConfiguration(
            report_dir=report_dir,
            statistics_backend="CSV"
        ),
        stopping=StoppingConfiguration(
            maximum_iterations=iterations,
            maximum_search_time=600
        ),
        search_algorithm=SearchAlgorithmConfiguration(
            population=50
        ),
        seeding=SeedingConfiguration(
            seed=42
        ),
        algorithm=Algorithm.MOSA
    )

    if use_semantics:
        config.enable_seed_examples = True
        config.use_docstring_semantics = True

    if verbose:
        # Set up verbose logging in Pynguin
        os.environ["PYNGUIN_LOG_LEVEL"] = "DEBUG"
        print(f"  Running in verbose mode with configuration:")
        for field in dataclasses.fields(config):
            field_name = field.name
            field_value = getattr(config, field_name)
            print(f"    {field_name}: {field_value}")
    else:
        print(f"  Executing pynguin with configuration: {config}")

    try:
        # Run pynguin
        set_configuration(config)
        result = execute_pynguin()
        return result
    except Exception as e:
        print(f"  Exception running pynguin: {e}")
        return None

def extract_coverage_data(stats_file):
    """Extract coverage data from the statistics CSV file."""
    if os.path.exists(stats_file):
        try:
            df = pd.read_csv(stats_file)
            # Extract final coverage values (last row)
            coverage_data = {}
            for col in df.columns:
                if 'Coverage' in col:
                    coverage_data[col.lower()] = df[col].iloc[-1]
            return coverage_data
        except Exception as e:
            print(f"  Error processing statistics file: {e}")
    else:
        print(f"  Statistics file not found: {stats_file}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Run Pynguin with and without semantics")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs for each configuration")
    parser.add_argument("--iterations", type=int, default=200, help="Maximum iterations for Pynguin")
    parser.add_argument("--project-path", type=str, default="tmp/evaluation/examples", help="Path to project")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    NUM_RUNS = args.runs
    PROJECT_PATH = args.project_path
    MAX_ITERATIONS = args.iterations
    VERBOSE = args.verbose

    print("Note: This script sets PYNGUIN_DANGER_AWARE=1 to allow Pynguin to run.")
    print("      See https://pynguin.readthedocs.io/en/latest/user/quickstart.html for details.")

    # Set environment variable
    os.environ["PYNGUIN_DANGER_AWARE"] = "1"

    # Verify pynguin is available
    try:
        import pynguin
        print(f"Found pynguin version: {pynguin.__version__}")
    except ImportError:
        print("Error: pynguin module not found. Make sure it's installed.")
        sys.exit(1)

    # Create timestamp for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_BASE_PATH = f"pynguin_experiment_{timestamp}"
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

    # Find all Python files in the project directory
    project_dir = Path(PROJECT_PATH)
    example_modules = [
        os.path.splitext(f.name)[0]
        for f in project_dir.glob("*.py")
        if f.is_file()
    ]

    print(f"Found {len(example_modules)} modules: {example_modules}")
    print(f"Running {NUM_RUNS} times for each module with and without semantics")
    print(f"Output will be saved to: {OUTPUT_BASE_PATH}")

    # Prepare data collection
    all_results = []

    for module_name in example_modules:
        print(f"\n--- Testing module: {module_name} ---")

        # Run without semantics
        for run in range(1, NUM_RUNS + 1):
            print(f"Standard run {run}/{NUM_RUNS}")
            output_dir = os.path.join(OUTPUT_BASE_PATH, f"{module_name}_standard_run{run}")
            os.makedirs(output_dir, exist_ok=True)

            start_time = time.time()
            result = run_pynguin(PROJECT_PATH, output_dir, module_name,
                                 use_semantics=False, iterations=MAX_ITERATIONS, verbose=VERBOSE)
            execution_time = time.time() - start_time

            if result is None:
                print(f"  Failed to run pynguin for {module_name} (standard)")
                continue

            # Collect coverage data
            stats_file = os.path.join(output_dir, "pynguin-report", "statistics.csv")
            coverage_data = extract_coverage_data(stats_file)

            if coverage_data:
                coverage_data.update({
                    'module': module_name,
                    'run': run,
                    'semantics': False,
                    'execution_time': execution_time
                })
                all_results.append(coverage_data)
            else:
                print(f"  No coverage data collected for {module_name} (standard, run {run})")

        # Run with semantics
        for run in range(1, NUM_RUNS + 1):
            print(f"Semantic run {run}/{NUM_RUNS}")
            output_dir = os.path.join(OUTPUT_BASE_PATH, f"{module_name}_semantic_run{run}")
            os.makedirs(output_dir, exist_ok=True)

            start_time = time.time()
            result = run_pynguin(PROJECT_PATH, output_dir, module_name,
                                 use_semantics=True, iterations=MAX_ITERATIONS, verbose=VERBOSE)
            execution_time = time.time() - start_time

            if result is None:
                print(f"  Failed to run pynguin for {module_name} (semantic)")
                continue

            # Collect coverage data
            stats_file = os.path.join(output_dir, "pynguin-report", "statistics.csv")
            coverage_data = extract_coverage_data(stats_file)

            if coverage_data:
                coverage_data.update({
                    'module': module_name,
                    'run': run,
                    'semantics': True,
                    'execution_time': execution_time
                })
                all_results.append(coverage_data)
            else:
                print(f"  No coverage data collected for {module_name} (semantic, run {run})")

    # Save all results to a CSV file
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = os.path.join(OUTPUT_BASE_PATH, "coverage_comparison.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")

        # Generate summary statistics
        summary_file = os.path.join(OUTPUT_BASE_PATH, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("EXPERIMENT SUMMARY\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modules tested: {', '.join(example_modules)}\n")
            f.write(f"Number of runs: {NUM_RUNS}\n\n")

            for module in results_df['module'].unique():
                f.write(f"\nModule: {module}\n")
                f.write("-" * 40 + "\n")

                standard = results_df[(results_df['module'] == module) & (results_df['semantics'] == False)]
                semantic = results_df[(results_df['module'] == module) & (results_df['semantics'] == True)]

                # Add execution time comparison
                std_time_avg = standard['execution_time'].mean() if not standard.empty else 0
                sem_time_avg = semantic['execution_time'].mean() if not semantic.empty else 0
                time_diff = sem_time_avg - std_time_avg
                time_percentage = (time_diff / std_time_avg * 100) if std_time_avg > 0 else float('inf')

                f.write(f"execution_time (seconds):\n")
                f.write(f"  Standard: {std_time_avg:.2f}\n")
                f.write(f"  Semantic: {sem_time_avg:.2f}\n")
                f.write(f"  Difference: {time_diff:.2f} ({time_percentage:.2f}%)\n\n")

                coverage_cols = [col for col in results_df.columns if 'coverage' in col]
                for col in coverage_cols:
                    std_avg = standard[col].mean() if not standard.empty else 0
                    sem_avg = semantic[col].mean() if not semantic.empty else 0
                    improvement = sem_avg - std_avg
                    rel_imp = (improvement / std_avg * 100) if std_avg > 0 else float('inf')

                    f.write(f"{col}:\n")
                    f.write(f"  Standard: {std_avg:.4f}\n")
                    f.write(f"  Semantic: {sem_avg:.4f}\n")
                    f.write(f"  Improvement: {improvement:.4f} ({rel_imp:.2f}%)\n\n")

        print(f"Summary written to {summary_file}")
    else:
        print("No results were collected!")

if __name__ == "__main__":
    main()
