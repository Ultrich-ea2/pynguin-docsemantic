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
import numpy as np

from pynguin.configuration import CoverageMetric, Minimization, MinimizationStrategy


def run_pynguin(project_path, output_path, module_name, run_iteration, use_semantics=False, iterations=100, verbose=False):
    """Run Pynguin with the specified parameters."""
    # Configure logging

    if verbose:
        logging.basicConfig(
            level=logging.INFO if verbose else logging.INFO,
            format='%(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger("pynguin")
        logger.setLevel(logging.INFO if verbose else logging.INFO)

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
    from pynguin.utils.statistics.runtimevariable import RuntimeVariable
    import dataclasses

    # Create configuration
    config = Configuration(
        project_path=project_path,
        module_name=module_name,
        test_case_output=TestCaseOutputConfiguration(
            output_path=output_path,
            assertion_generation="MUTATION_ANALYSIS",
            minimization=Minimization(
                test_case_minimization_strategy=MinimizationStrategy.NONE
            )
        ),
        statistics_output=StatisticsOutputConfiguration(
            report_dir=report_dir,
            statistics_backend="CSV",
            coverage_metrics=[CoverageMetric("BRANCH")], #, CoverageMetric("LINE")
            output_variables=[
                # Basic tracking
                RuntimeVariable.TargetModule,
                RuntimeVariable.TotalTime,
                RuntimeVariable.SearchTime,
                RuntimeVariable.AlgorithmIterations,

                # Coverage metrics
                RuntimeVariable.Coverage,
                # RuntimeVariable.LineCoverage,
                RuntimeVariable.BranchCoverage,
                RuntimeVariable.FinalBranchCoverage,

                # Test suite metrics
                RuntimeVariable.Size,
                RuntimeVariable.Length,
                RuntimeVariable.FinalSize,
                RuntimeVariable.FinalLength,
                RuntimeVariable.Fitness,

                # Timeline data for convergence
                # RuntimeVariable.CoverageTimeline,
                # RuntimeVariable.SizeTimeline,
                # RuntimeVariable.LengthTimeline,
                # RuntimeVariable.FitnessTimeline,

                # Code complexity
                RuntimeVariable.Goals,
                RuntimeVariable.Predicates,
                RuntimeVariable.Lines,
                RuntimeVariable.McCabeAST,
                RuntimeVariable.AccessibleObjectsUnderTest,
            ],
        ),
        stopping=StoppingConfiguration(
            maximum_iterations=iterations,
            maximum_search_time=300 # 5 minutes
        ),
        search_algorithm=SearchAlgorithmConfiguration(
            population=50
        ),
        seeding = SeedingConfiguration(
            seed=int.from_bytes(hashlib.sha256(str(run_iteration).encode()).digest()[:4], 'big')
        ),
        algorithm=Algorithm.DYNAMOSA,
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
    except Exception as e:
        print(f"  Exception running pynguin: {e}")
        return None


def extract_coverage_data(stats_file, timeline_file=None):
    """Extract coverage data from the statistics CSV file."""
    coverage_data = {}

    if os.path.exists(stats_file):
        try:
            stats_df = pd.read_csv(stats_file)
            if not stats_df.empty:
                # Normalize column names to lowercase for consistency
                stats_df.columns = stats_df.columns.str.lower()

                # Extract the last row (most recent values)
                last_row = stats_df.iloc[-1]

                # Map the normalized column names
                for col in stats_df.columns:
                    coverage_data[col] = last_row[col]

        except Exception as e:
            print(f"  Error reading statistics file {stats_file}: {e}")

    timeline_data = None
    if timeline_file and os.path.exists(timeline_file):
        try:
            timeline_df = pd.read_csv(timeline_file)
            if not timeline_df.empty:
                timeline_df.columns = timeline_df.columns.str.lower()
                timeline_data = timeline_df
        except Exception as e:
            print(f"  Error reading timeline file {timeline_file}: {e}")

    return coverage_data, timeline_data

def main():
    parser = argparse.ArgumentParser(description="Run Pynguin with and without semantics")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs for each configuration")
    parser.add_argument("--iterations", type=int, default=250, help="Maximum iterations for Pynguin")
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

    # Create timestamp for the evaluation output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_BASE_PATH = f"pynguin_evaluation_{timestamp}"
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
    timeline_results = []  # Add this to collect timeline data

    for module_name in example_modules:
        print(f"\n--- Testing module: {module_name} ---")

        run_without_semantics(MAX_ITERATIONS, NUM_RUNS, OUTPUT_BASE_PATH, PROJECT_PATH, VERBOSE, all_results,
                              timeline_results, module_name)

        run_with_semantics(MAX_ITERATIONS, NUM_RUNS, OUTPUT_BASE_PATH, PROJECT_PATH, VERBOSE, all_results, timeline_results, module_name)

    # Save all results to a CSV file
    if all_results:
        save_all_results(NUM_RUNS, OUTPUT_BASE_PATH, all_results, example_modules)

        # Generate comprehensive analysis and detailed summary
        analyze_comprehensive_results(all_results, timeline_results, OUTPUT_BASE_PATH)
        generate_detailed_summary(all_results, OUTPUT_BASE_PATH)

        print(f"\nComprehensive analysis plots saved to {OUTPUT_BASE_PATH}")
        print(f"Detailed summary saved to {OUTPUT_BASE_PATH}/detailed_summary.txt")
    else:
        print("No results were collected!")


# def save_all_results(NUM_RUNS, OUTPUT_BASE_PATH, all_results, example_modules):
#     results_df = pd.DataFrame(all_results)
#     results_file = os.path.join(OUTPUT_BASE_PATH, "coverage_comparison.csv")
#     results_df.to_csv(results_file, index=False)
#     print(f"\nResults saved to {results_file}")
#
#     # Generate summary statistics
#     summary_file = os.path.join(OUTPUT_BASE_PATH, "summary.txt")
#     with open(summary_file, 'w') as f:
#         f.write("EVALUATION SUMMARY\n")
#         f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Modules tested: {', '.join(example_modules)}\n")
#         f.write(f"Number of runs: {NUM_RUNS}\n\n")
#
#         for module in results_df['module'].unique():
#             f.write(f"\nModule: {module}\n")
#             f.write("-" * 40 + "\n")
#
#             standard = results_df[(results_df['module'] == module) & (results_df['semantics'] == False)]
#             semantic = results_df[(results_df['module'] == module) & (results_df['semantics'] == True)]
#
#             # Add execution time comparison
#             std_time_avg = standard['execution_time'].mean() if not standard.empty else 0
#             sem_time_avg = semantic['execution_time'].mean() if not semantic.empty else 0
#             time_diff = sem_time_avg - std_time_avg
#             time_percentage = (time_diff / std_time_avg * 100) if std_time_avg > 0 else float('inf')
#
#             f.write(f"execution_time (seconds):\n")
#             f.write(f"  Standard: {std_time_avg:.2f}\n")
#             f.write(f"  Semantic: {sem_time_avg:.2f}\n")
#             f.write(f"  Difference: {time_diff:.2f} ({time_percentage:.2f}%)\n\n")
#
#             coverage_cols = [col for col in results_df.columns if 'coverage' in col]
#             for col in coverage_cols:
#                 std_avg = standard[col].mean() if not standard.empty else 0
#                 sem_avg = semantic[col].mean() if not semantic.empty else 0
#                 improvement = sem_avg - std_avg
#                 rel_imp = (improvement / std_avg * 100) if std_avg > 0 else float('inf')
#
#                 f.write(f"{col}:\n")
#                 f.write(f"  Standard: {std_avg:.4f}\n")
#                 f.write(f"  Semantic: {sem_avg:.4f}\n")
#                 f.write(f"  Improvement: {improvement:.4f} ({rel_imp:.2f}%)\n\n")
#     print(f"Summary written to {summary_file}")


def run_without_semantics(MAX_ITERATIONS, NUM_RUNS, OUTPUT_BASE_PATH, PROJECT_PATH, VERBOSE, all_results,
                          timeline_results, module_name):
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

        # Collect both final and timeline coverage data
        stats_file = os.path.join(output_dir, "pynguin-report", "statistics.csv")
        timeline_file = os.path.join(output_dir, "pynguin-report", "timeline.csv")

        coverage_data, timeline_data = extract_coverage_data(stats_file, timeline_file)

        if coverage_data:
            coverage_data.update({
                'module': module_name,
                'run': run,
                'semantics': False,
                'execution_time': execution_time,
            })
            all_results.append(coverage_data)

        if timeline_data is not None:
            timeline_data['module'] = module_name
            timeline_data['run'] = run
            timeline_data['semantics'] = False
            timeline_results.append(timeline_data)
        else:
            print(f"  No coverage data collected for {module_name} (standard, run {run})")


def run_with_semantics(MAX_ITERATIONS, NUM_RUNS, OUTPUT_BASE_PATH, PROJECT_PATH, VERBOSE, all_results, timeline_results,
                       module_name):
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

        # Collect both final and timeline coverage data
        stats_file = os.path.join(output_dir, "pynguin-report", "statistics.csv")
        timeline_file = os.path.join(output_dir, "pynguin-report", "timeline.csv")

        coverage_data, timeline_data = extract_coverage_data(stats_file, timeline_file)

        if coverage_data:
            coverage_data.update({
                'module': module_name,
                'run': run,
                'semantics': True,
                'execution_time': execution_time,
            })
            all_results.append(coverage_data)

        if timeline_data is not None:
            timeline_data['module'] = module_name
            timeline_data['run'] = run
            timeline_data['semantics'] = True
            timeline_results.append(timeline_data)
        else:
            print(f"  No coverage data collected for {module_name} (semantic, run {run})")


def analyze_comprehensive_results(all_results, timeline_results, output_path):
    """Perform comprehensive analysis of results."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not all_results:
        return

    df = pd.DataFrame(all_results)

    # 1. Coverage vs Complexity Analysis
    plt.figure(figsize=(12, 8))
    complexity_metrics = ['goals', 'predicates', 'lines', 'mccabeast']

    for i, metric in enumerate(complexity_metrics, 1):
        if metric in df.columns:
            plt.subplot(2, 2, i)
            for semantics in [False, True]:
                subset = df[df['semantics'] == semantics]
                if not subset.empty:
                    plt.scatter(subset[metric], subset['coverage'],
                                alpha=0.6, label=f"Semantic: {semantics}")
            plt.xlabel(metric.title())
            plt.ylabel('Coverage')
            plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'coverage_vs_complexity.png'), dpi=300)
    plt.close()

    # 2. Efficiency Analysis (Coverage per Time)
    if 'searchtime' in df.columns:
        df['efficiency'] = df['coverage'] / df['searchtime']

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='semantics', y='efficiency')
        plt.title('Coverage Efficiency (Coverage/SearchTime)')
        plt.savefig(os.path.join(output_path, 'efficiency_analysis.png'), dpi=300)
        plt.close()

    # 3. Test Suite Quality Analysis
    quality_metrics = ['size', 'length', 'finalsize', 'finallength']
    existing_metrics = [m for m in quality_metrics if m in df.columns]

    if existing_metrics:
        fig, axes = plt.subplots(1, len(existing_metrics), figsize=(15, 5))
        if len(existing_metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(existing_metrics):
            sns.boxplot(data=df, x='semantics', y=metric, ax=axes[i])
            axes[i].set_title(f'Test Suite {metric.title()}')

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'test_suite_quality.png'), dpi=300)
        plt.close()


def generate_detailed_summary(all_results, output_path):
    """Generate detailed summary with additional metrics."""
    df = pd.DataFrame(all_results)

    summary_file = os.path.join(output_path, "detailed_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("DETAILED EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        for module in df['module'].unique():
            module_data = df[df['module'] == module]
            f.write(f"Module: {module}\n")
            f.write("-" * 30 + "\n")

            for semantics in [False, True]:
                subset = module_data[module_data['semantics'] == semantics]
                if subset.empty:
                    continue

                mode = "Semantic" if semantics else "Standard"
                f.write(f"\n{mode} Mode:\n")

                # Basic stats for all available metrics
                numeric_cols = subset.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['run', 'semantics']:
                        mean_val = subset[col].mean()
                        std_val = subset[col].std()
                        f.write(f"  {col}: {mean_val:.4f} ± {std_val:.4f}\n")

            f.write("\n" + "=" * 30 + "\n")


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

            # Timing metrics
            f.write("=== TIMING METRICS ===\n")
            if 'totaltime' in results_df.columns:
                std_total = standard[
                                'totaltime'].mean() / 1e9 if not standard.empty else 0  # Convert nanoseconds to seconds
                sem_total = semantic['totaltime'].mean() / 1e9 if not semantic.empty else 0
                f.write(f"Total Time (seconds):\n")
                f.write(f"  Standard: {std_total:.2f} ± {standard['totaltime'].std() / 1e9:.2f}\n")
                f.write(f"  Semantic: {sem_total:.2f} ± {semantic['totaltime'].std() / 1e9:.2f}\n")
                f.write(f"  Difference: {sem_total - std_total:.2f} seconds\n\n")

            if 'searchtime' in results_df.columns:
                std_search = standard['searchtime'].mean() / 1e9 if not standard.empty else 0
                sem_search = semantic['searchtime'].mean() / 1e9 if not semantic.empty else 0
                f.write(f"Search Time (seconds):\n")
                f.write(f"  Standard: {std_search:.2f} ± {standard['searchtime'].std() / 1e9:.2f}\n")
                f.write(f"  Semantic: {sem_search:.2f} ± {semantic['searchtime'].std() / 1e9:.2f}\n")
                f.write(f"  Difference: {sem_search - std_search:.2f} seconds\n\n")

            # Algorithm metrics
            f.write("=== ALGORITHM METRICS ===\n")
            if 'algorithmiterations' in results_df.columns:
                std_iter = standard['algorithmiterations'].mean() if not standard.empty else 0
                sem_iter = semantic['algorithmiterations'].mean() if not semantic.empty else 0
                f.write(f"Algorithm Iterations:\n")
                f.write(f"  Standard: {std_iter:.1f} ± {standard['algorithmiterations'].std():.1f}\n")
                f.write(f"  Semantic: {sem_iter:.1f} ± {semantic['algorithmiterations'].std():.1f}\n")
                f.write(f"  Difference: {sem_iter - std_iter:.1f}\n\n")

            # Coverage metrics
            f.write("=== COVERAGE METRICS ===\n")
            coverage_metrics = ['coverage', 'linecoverage', 'branchcoverage', 'finalbranchcoverage']
            for metric in coverage_metrics:
                if metric in results_df.columns:
                    std_avg = standard[metric].mean() if not standard.empty else 0
                    sem_avg = semantic[metric].mean() if not semantic.empty else 0
                    improvement = sem_avg - std_avg
                    rel_imp = (improvement / std_avg * 100) if std_avg > 0 else float('inf')

                    f.write(f"{metric.replace('coverage', ' Coverage').title()}:\n")
                    f.write(f"  Standard: {std_avg:.4f} ± {standard[metric].std():.4f}\n")
                    f.write(f"  Semantic: {sem_avg:.4f} ± {semantic[metric].std():.4f}\n")
                    f.write(f"  Improvement: {improvement:.4f} ({rel_imp:.2f}%)\n\n")

            # Test suite metrics
            f.write("=== TEST SUITE METRICS ===\n")
            suite_metrics = ['size', 'length', 'finalsize', 'finallength']
            for metric in suite_metrics:
                if metric in results_df.columns:
                    std_avg = standard[metric].mean() if not standard.empty else 0
                    sem_avg = semantic[metric].mean() if not semantic.empty else 0
                    f.write(f"{metric.replace('final', 'Final ').title()}:\n")
                    f.write(f"  Standard: {std_avg:.1f} ± {standard[metric].std():.1f}\n")
                    f.write(f"  Semantic: {sem_avg:.1f} ± {semantic[metric].std():.1f}\n")
                    f.write(f"  Difference: {sem_avg - std_avg:.1f}\n\n")

            # Code complexity metrics
            f.write("=== CODE COMPLEXITY METRICS ===\n")
            complexity_metrics = ['goals', 'predicates', 'lines', 'mccabeast', 'accessibleobjectsundertest']
            for metric in complexity_metrics:
                if metric in results_df.columns:
                    # These should be the same for both approaches, so just show one value
                    value = standard[metric].iloc[0] if not standard.empty else (
                        semantic[metric].iloc[0] if not semantic.empty else 0)
                    f.write(
                        f"{metric.replace('mccabeast', 'McCabe Complexity').replace('accessibleobjectsundertest', 'Accessible Objects').title()}: {value}\n")

            f.write("\n" + "=" * 50 + "\n")

    print(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()
