import subprocess
import os
import sys
import pandas as pd
import time
from datetime import datetime
import argparse
from pathlib import Path
import logging
import hashlib
import importlib.util

from pynguin.configuration import CoverageMetric


def find_python_modules(project_path, exclude_patterns=None):
    """Find all Python modules in a project, excluding test files and __pycache__."""
    if exclude_patterns is None:
        exclude_patterns = ['test_', '_test', '__pycache__', '.git', '.pytest_cache', 'build', 'dist']

    modules = []
    project_dir = Path(project_path)

    # First, find all __init__.py files to identify package structure
    packages = set()
    for init_file in project_dir.rglob("__init__.py"):
        relative_path = init_file.relative_to(project_dir)
        package_parts = list(relative_path.parts[:-1])  # Remove __init__.py
        if package_parts:  # Only if not in root
            package_name = '.'.join(package_parts)
            packages.add(package_name)

    for py_file in project_dir.rglob("*.py"):
        # Skip files that match exclude patterns
        if any(pattern in str(py_file) for pattern in exclude_patterns):
            continue

        # Skip __init__.py files - they define packages, not modules
        if py_file.name == '__init__.py':
            continue

        # Convert file path to module name
        relative_path = py_file.relative_to(project_dir)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]

        # Check if this file is part of a valid package structure
        if len(module_parts) > 1:
            # For nested modules, check if parent directories have __init__.py
            parent_package = '.'.join(module_parts[:-1])
            if parent_package not in packages:
                # Skip modules that aren't in proper packages
                continue

        module_name = '.'.join(module_parts)

        # Skip modules with empty parts (can happen with malformed paths)
        if any(not part for part in module_parts):
            continue

        modules.append(module_name)

    return sorted(modules)


def is_module_importable(project_path, module_name):
    """Check if a module can be imported without errors."""
    original_path = sys.path.copy()
    try:
        # Add project path to Python path
        if project_path not in sys.path:
            sys.path.insert(0, project_path)

        # Try to import the module
        module = importlib.import_module(module_name)

        # Additional validation: check if module has analyzable content
        if hasattr(module, '__file__') and module.__file__:
            # Only consider modules with actual Python source files
            if module.__file__.endswith(('.py', '.pyw')):
                return True
            else:
                print(f"  Module {module_name} is not a pure Python module: {module.__file__}")
                return False
        else:
            print(f"  Module {module_name} has no source file (built-in or C extension)")
            return False

    except Exception as e:
        print(f"  Module {module_name} not importable: {e}")
        return False
    finally:
        sys.path = original_path


def run_pynguin(project_path, output_path, module_name, run_iteration, use_semantics=False, iterations=100, verbose=True):
    """Run Pynguin with the specified parameters."""
    # Configure logging
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger("pynguin")
        logger.setLevel(logger.DEBUG if verbose else logging.INFO)

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

    # Ensure project path is absolute
    abs_project_path = os.path.abspath(project_path)

    # Create configuration
    config = Configuration(
        project_path=project_path,
        module_name=module_name,
        test_case_output=TestCaseOutputConfiguration(
            output_path=output_path,
            assertion_generation="MUTATION_ANALYSIS"
        ),
        statistics_output=StatisticsOutputConfiguration(
            report_dir=report_dir,
            statistics_backend="CSV",
            coverage_metrics=[CoverageMetric("BRANCH")], #, CoverageMetric("LINE")
        ),
        stopping=StoppingConfiguration(
            maximum_iterations=iterations,
            maximum_search_time=600 # 10 minutes
        ),
        search_algorithm=SearchAlgorithmConfiguration(
            population=50
        ),
        seeding = SeedingConfiguration(
            seed=int.from_bytes(hashlib.sha256(str(run_iteration).encode()).digest()[:4], 'big')
        ),
        algorithm=Algorithm.DYNAMOSA,
        # algorithm=Algorithm.MIO,
    )

    if use_semantics:
        config.enable_seed_examples = True
        config.use_docstring_semantics = True

    if verbose:
        # Set up verbose logging in Pynguin
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
    except AttributeError as e:
        if "'wrapper_descriptor' object has no attribute '__module__'" in str(e):
            print(f"  Skipping {module_name}: Contains non-introspectable objects (C extensions/built-ins)")
        else:
            print(f"  AttributeError in pynguin for {module_name}: {e}")
        return None
    except Exception as e:
        print(f"  Exception running pynguin on {module_name}: {type(e).__name__}: {e}")
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
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for each configuration")
    parser.add_argument("--iterations", type=int, default=250, help="Maximum iterations for Pynguin")
    parser.add_argument("--project-path", type=str, default="tmp/evaluation/examples", help="Path to project")
    parser.add_argument("--max-modules", type=int, default=None, help="Maximum number of modules to test")
    parser.add_argument("--exclude", nargs='+', default=['test_', '_test', '__pycache__', '.git'],
                        help="Patterns to exclude from module discovery")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    NUM_RUNS = args.runs
    PROJECT_PATH = args.project_path
    MAX_ITERATIONS = args.iterations
    VERBOSE = args.verbose
    MAX_MODULES = args.max_modules

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

    # Create timestamp for the evaluation output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_BASE_PATH = f"pynguin_evaluation_{timestamp}"
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

    # Find all Python modules in the repository
    print(f"Discovering modules in {PROJECT_PATH}...")
    all_modules = find_python_modules(PROJECT_PATH, args.exclude)
    print(f"Found {len(all_modules)} potential modules")

    # Filter to only importable modules
    print("Checking module importability...")
    example_modules = []
    for module in all_modules:
        if is_module_importable(PROJECT_PATH, module):
            example_modules.append(module)
            if MAX_MODULES and len(example_modules) >= MAX_MODULES:
                break

    print(f"Found {len(example_modules)} importable modules")
    if VERBOSE:
        for module in example_modules:
            print(f"  - {module}")

    if not example_modules:
        print("No importable modules found!")
        sys.exit(1)

    print(f"Running {NUM_RUNS} times for each module with and without semantics")
    print(f"Output will be saved to: {OUTPUT_BASE_PATH}")

    # Prepare data collection
    all_results = []

    for module_name in example_modules:
        print(f"\n--- Testing module: {module_name} ---")

        run_without_semantics(MAX_ITERATIONS, NUM_RUNS, OUTPUT_BASE_PATH, PROJECT_PATH, VERBOSE, all_results,
                              module_name)

        run_with_semantics(MAX_ITERATIONS, NUM_RUNS, OUTPUT_BASE_PATH, PROJECT_PATH, VERBOSE, all_results, module_name)

    # Save all results to a CSV file
    if all_results:
        save_all_results(NUM_RUNS, OUTPUT_BASE_PATH, all_results, example_modules)
    else:
        print("No results were collected!")


def save_all_results(NUM_RUNS, OUTPUT_BASE_PATH, all_results, example_modules):
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(OUTPUT_BASE_PATH, "coverage_comparison.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

    # Generate summary statistics
    summary_file = os.path.join(OUTPUT_BASE_PATH, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("EVALUATION SUMMARY\n")
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


def run_with_semantics(MAX_ITERATIONS, NUM_RUNS, OUTPUT_BASE_PATH, PROJECT_PATH, VERBOSE, all_results, module_name):
    for run in range(1, NUM_RUNS + 1):
        print(f"Semantic run {run}/{NUM_RUNS}")
        output_dir = os.path.join(OUTPUT_BASE_PATH, f"{module_name}_semantic_run{run}")
        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()
        result = run_pynguin(PROJECT_PATH, output_dir, module_name, run,
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


def run_without_semantics(MAX_ITERATIONS, NUM_RUNS, OUTPUT_BASE_PATH, PROJECT_PATH, VERBOSE, all_results, module_name):
    for run in range(1, NUM_RUNS + 1):
        print(f"Standard run {run}/{NUM_RUNS}")
        output_dir = os.path.join(OUTPUT_BASE_PATH, f"{module_name}_standard_run{run}")
        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()
        result = run_pynguin(PROJECT_PATH, output_dir, module_name, run,
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


if __name__ == "__main__":
    main()
