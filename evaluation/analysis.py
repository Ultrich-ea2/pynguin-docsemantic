# evaluation/analysis.py
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from evaluation import StatisticalResults

class AnalysisPlots:
    def __init__(self, output_path, pynguin_config):
        self.output_path = output_path
        self.run_config = pynguin_config

    def full_analysis(self, run_results, modules):
        # Process timeline data in chunks to reduce memory usage
        timeline_results = []
        for r in run_results:
            if r.timeline_data:
                timeline_results.append({
                    **r.timeline_data,
                    "module": r.module,
                    "run": r.run_id,
                    "semantics": r.semantics_enabled,
                    "execution_time": r.execution_time
                })

        if timeline_results:
            self.visualize_timelines(timeline_results)
            # Force garbage collection after timeline processing
            del timeline_results
            gc.collect()

        # Create results DataFrame only when needed
        all_results = [
            {**r.coverage_metrics, "module": r.module, "run": r.run_id, "semantics": r.semantics_enabled,
             "execution_time": r.execution_time}
            for r in run_results
        ]
        self.comprehensive_stats(all_results, modules)
        self.module_barplots(all_results)
        generate_summary_table(all_results, self.output_path)

    def visualize_timelines(self, timeline_results):
        if not timeline_results:
            print("No timeline data available for visualization.")
            return

        timeline_df = pd.DataFrame(timeline_results)
        vis_dir = os.path.join(self.output_path, 'timeline_visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Process modules one at a time to reduce memory usage
        for module in timeline_df['module'].unique():
            module_data = timeline_df[timeline_df['module'] == module].copy()
            _plot_module_timelines(module_data, module, vis_dir, self.run_config)
            # Clean up module data immediately
            del module_data
            gc.collect()

        _plot_overall_timelines(timeline_df, vis_dir)
        _plot_convergence_analysis(timeline_df, vis_dir)

        # Clean up timeline DataFrame
        del timeline_df
        gc.collect()

        print(f"Timeline visualizations saved to {vis_dir}")

    def comprehensive_stats(self, all_results, modules):
        if not all_results:
            print("No final results to analyze.")
            return

        df = pd.DataFrame(all_results)

        # Process each visualization separately and clean up

        _plot_efficiency_boxplot(df, self.output_path)
        gc.collect()

        _plot_test_suite_quality(df, self.output_path)
        gc.collect()

    def module_barplots(self, all_results):
        """Generate module-wise barplots comparing coverage and execution time for standard and semantic runs."""
        df = pd.DataFrame(all_results)
        if "module" not in df.columns or "semantics" not in df.columns:
            print("Cannot plot module barplots: missing required data.")
            return

        # Coverage barplot
        _plot_module_metric_barplot(
            df,
            metric="coverage",
            ylabel="Coverage",
            title="Coverage per Module (Standard vs. Semantic)",
            filename="module_coverage_barplot.png",
            output_path=self.output_path
        )
        gc.collect()

        # Execution time barplot
        _plot_module_metric_barplot(
            df,
            metric="execution_time",
            ylabel="Generation Time (seconds)",
            title="Generation Time per Module (Standard vs. Semantic)",
            filename="module_generation_time_barplot.png",
            output_path=self.output_path
        )
        gc.collect()

def _plot_module_timelines(module_data, module_name, vis_dir, pynguin_config):
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metric_map = {
        'Coverage': 'coveragetimeline',
        'Test Suite Size': 'sizetimeline',
        'Test Suite Length': 'lengthtimeline',
        'Fitness': 'fitnesstimeline'
    }

    for idx, (metric_name, base_col) in enumerate(metric_map.items()):
        if idx >= 4:
            continue

        cols = [col for col in module_data.columns if col.startswith(base_col)]
        cols = sorted(cols, key=lambda x: int(x.split('_t')[-1]))

        if not cols:
            continue

        ax = axes[idx]
        colors = ['#2E86AB', '#A23B72']
        linestyles = ['-', '--']
        alphas = [0.8, 0.8]
        fill_alphas = [0.15, 0.15]

        # Track if any data was plotted
        has_data = False

        for i, (semantics, color, linestyle, alpha, fill_alpha) in enumerate(
            zip([False, True], colors, linestyles, alphas, fill_alphas)):
            subset = module_data[module_data['semantics'] == semantics]
            if subset.empty:
                continue

            # Collect all timeline data for this configuration
            all_runs_data = []

            for _, row in subset.iterrows():
                # Get actual execution time (prefer SearchTime, fallback to execution_time)
                if 'SearchTime' in row and pd.notna(row['SearchTime']):
                    total_time = row['SearchTime']
                elif 'execution_time' in row and pd.notna(row['execution_time']):
                    total_time = row['execution_time']
                else:
                    continue

                # Extract timeline values for this run
                timeline_vals = []
                for col in cols:
                    if col in row and pd.notna(row[col]):
                        timeline_vals.append(float(row[col]))
                    else:
                        break  # Stop at first missing value

                if len(timeline_vals) < 2:  # Need at least 2 points
                    continue

                # Find last changing point for this run (use more lenient threshold)
                last_change_idx = len(timeline_vals) - 1  # Default to full length
                change_threshold = 1e-8  # More lenient threshold

                for j in range(len(timeline_vals) - 1, 0, -1):  # Search backwards
                    if abs(timeline_vals[j] - timeline_vals[j-1]) > change_threshold:
                        last_change_idx = j
                        break

                # Ensure we have at least 10 points or 50% of the data
                min_points = max(10, int(0.5 * len(timeline_vals)))
                trim_idx = max(min_points, last_change_idx + 1)
                trim_idx = min(trim_idx, len(timeline_vals))

                trimmed_vals = timeline_vals[:trim_idx]

                # Create time points based on actual execution time
                if len(trimmed_vals) > 1:
                    time_points = [total_time * (j / (len(trimmed_vals) - 1)) for j in range(len(trimmed_vals))]
                    all_runs_data.append((time_points, trimmed_vals))

            if not all_runs_data:
                continue

            # Create common time grid based on all runs
            max_time = max(time_points[-1] for time_points, _ in all_runs_data)
            time_grid = np.linspace(0, max_time, 100)

            # Interpolate each run to the common time grid
            interpolated_values = []
            for time_points, values in all_runs_data:
                if len(time_points) >= 2:  # Need at least 2 points for interpolation
                    interpolated = np.interp(time_grid, time_points, values)
                    interpolated_values.append(interpolated)

            if interpolated_values:
                # Calculate mean and std across runs
                interpolated_array = np.array(interpolated_values)
                means = np.nanmean(interpolated_array, axis=0)
                stds = np.nanstd(interpolated_array, axis=0)

                label = f"{'Semantic' if semantics else 'Standard'} (n={len(interpolated_values)})"

                # Plot line with transparency and different styles
                ax.plot(time_grid, means, label=label, color=color, linewidth=2.5,
                        linestyle=linestyle, alpha=alpha, zorder=3 - i)

                # Add fill with lower transparency
                ax.fill_between(time_grid, means - stds, means + stds, alpha=fill_alpha,
                                color=color, zorder=1 - i)

                has_data = True

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Timeline - {module_name}')

        # Only add legend if data was plotted
        if has_data:
            ax.legend(loc='best', framealpha=0.9)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{module_name}_timeline_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Force cleanup
    del fig, axes
    gc.collect()

def generate_summary_table(all_results, output_path):
    """
    Generate comprehensive summary tables comparing standard vs semantic configurations per module.
    Includes statistical analysis and improvement metrics.
    """
    import pandas as pd

    df = pd.DataFrame(all_results)

    # Define metrics to include in summary
    metrics = {
        'coverage': 'Coverage (%)',
        'execution_time': 'Execution Time (s)'
    }

    # Filter for available metrics
    available_metrics = {k: v for k, v in metrics.items() if k in df.columns}

    if not available_metrics:
        print("No metrics available for summary table.")
        return

    # Generate basic summary table (existing functionality)
    summary_df = _generate_basic_summary_table(df, available_metrics, output_path)

    # Generate scientific comparison table
    scientific_df = _generate_scientific_comparison_table(df, available_metrics, output_path)

    # Generate statistical test results
    statistical_df = _generate_statistical_tests_table(df, available_metrics, output_path)

    # Generate improvement metrics table
    improvement_df = _generate_improvement_metrics_table(df, available_metrics, output_path)

    print(f"All summary tables saved to {output_path}")
    return summary_df, scientific_df, statistical_df, improvement_df


def _generate_basic_summary_table(df, available_metrics, output_path):
    """Generate the basic summary table (original functionality)"""
    summary_data = []

    for module in sorted(df['module'].unique()):
        module_df = df[df['module'] == module]

        for semantics in [False, True]:
            config_df = module_df[module_df['semantics'] == semantics]

            if config_df.empty:
                continue

            row = {
                'Module': module,
                'Configuration': 'Standard' if not semantics else 'Semantic',
                'Runs': len(config_df)
            }

            for metric_key, metric_name in available_metrics.items():
                values = config_df[metric_key].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    row[f'Ø {metric_name}'] = f"{mean_val:.2f}"
                    row[f'σ {metric_name}'] = f"{std_val:.2f}"
                else:
                    row[f'Ø {metric_name}'] = "N/A"
                    row[f'σ {metric_name}'] = "N/A"

            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Save basic summary
    csv_path = os.path.join(output_path, 'summary_table.csv')
    summary_df.to_csv(csv_path, index=False)

    txt_path = os.path.join(output_path, 'summary_table.txt')
    with open(txt_path, 'w') as f:
        f.write("CONFIGURATION COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        f.write("Legend:\n")
        f.write("Ø = Mean (average)\n")
        f.write("σ = Standard deviation\n")

    return summary_df


def _generate_scientific_comparison_table(df, available_metrics, output_path):
    """Generate scientific comparison table with improvement values and percentages"""
    comparison_data = []

    for module in sorted(df['module'].unique()):
        module_df = df[df['module'] == module]

        std_df = module_df[module_df['semantics'] == False]
        sem_df = module_df[module_df['semantics'] == True]

        if std_df.empty or sem_df.empty:
            continue

        row = {'Module': module}

        for metric_key, metric_name in available_metrics.items():
            std_values = std_df[metric_key].dropna()
            sem_values = sem_df[metric_key].dropna()

            if len(std_values) > 0 and len(sem_values) > 0:
                std_mean = std_values.mean()
                sem_mean = sem_values.mean()

                # Calculate improvement (absolute difference)
                improvement = sem_mean - std_mean
                improvement_std = np.sqrt(std_values.var() + sem_values.var())

                # Calculate percentage improvement
                if std_mean != 0:
                    improvement_pct = (improvement / std_mean) * 100
                else:
                    improvement_pct = 0

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(std_values) - 1) * std_values.var() +
                                      (len(sem_values) - 1) * sem_values.var()) /
                                     (len(std_values) + len(sem_values) - 2))
                cohens_d = improvement / pooled_std if pooled_std > 0 else 0

                row[f'{metric_name} Improvement'] = f"{improvement:.3f} ± {improvement_std:.3f}"
                row[f'{metric_name} Improvement (%)'] = f"{improvement_pct:.1f}%"
                row[f'{metric_name} Effect Size (d)'] = f"{cohens_d:.3f}"
            else:
                row[f'{metric_name} Improvement'] = "N/A"
                row[f'{metric_name} Improvement (%)'] = "N/A"
                row[f'{metric_name} Effect Size (d)'] = "N/A"

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Save scientific comparison
    csv_path = os.path.join(output_path, 'scientific_comparison_table.csv')
    comparison_df.to_csv(csv_path, index=False)

    txt_path = os.path.join(output_path, 'scientific_comparison_table.txt')
    with open(txt_path, 'w') as f:
        f.write("SCIENTIFIC COMPARISON TABLE\n")
        f.write("=" * 100 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        f.write("Legend:\n")
        f.write("Improvement = Semantic Mean - Standard Mean\n")
        f.write("Improvement (%) = (Improvement / Standard Mean) × 100\n")
        f.write("Effect Size (d) = Cohen's d (standardized effect size)\n")
        f.write("  Small: 0.2, Medium: 0.5, Large: 0.8\n")

    return comparison_df


def _generate_statistical_tests_table(df, available_metrics, output_path):
    """Generate statistical tests table with means, std, t-tests organized by module"""
    statistical_data = []

    for module in sorted(df['module'].unique()):
        module_df = df[df['module'] == module]

        std_df = module_df[module_df['semantics'] == False]
        sem_df = module_df[module_df['semantics'] == True]

        if std_df.empty or sem_df.empty:
            continue

        # Add metrics available in the data beyond the basic ones
        all_metrics = {
            'coverage': 'Coverage',
            'size': 'Size',
            'length': 'Length',
            'fitness': 'Fitness',
            'execution_time': 'Execution Time'
        }

        # Filter for metrics that exist in the data
        available_module_metrics = {k: v for k, v in all_metrics.items() if k in module_df.columns}

        for metric_key, metric_name in available_module_metrics.items():
            std_values = std_df[metric_key].dropna()
            sem_values = sem_df[metric_key].dropna()

            if len(std_values) >= 2 and len(sem_values) >= 2:
                # Calculate means and standard deviations
                std_mean = std_values.mean()
                std_std = std_values.std()
                sem_mean = sem_values.mean()
                sem_std = sem_values.std()

                # Perform t-test (independent samples)
                t_stat, p_value = stats.ttest_ind(sem_values, std_values, equal_var=False)

                # Determine significance
                significant = "YES" if p_value < 0.05 else "NO"

                # Create row for this module-metric combination
                row = {
                    'Module': module,
                    'Metric': metric_name,
                    'Standard Mean': f"{std_mean:.4f}",
                    'Standard Std': f"{std_std:.4f}",
                    'Semantic Mean': f"{sem_mean:.4f}",
                    'Semantic Std': f"{sem_std:.4f}",
                    't-statistic': f"{t_stat:.4f}",
                    'p-value': f"{p_value:.4f}",
                    'Significant': significant
                }

                statistical_data.append(row)

    statistical_df = pd.DataFrame(statistical_data)

    # Save statistical tests
    csv_path = os.path.join(output_path, 'statistical_tests_table.csv')
    statistical_df.to_csv(csv_path, index=False)

    txt_path = os.path.join(output_path, 'statistical_tests_table.txt')
    with open(txt_path, 'w') as f:
        f.write("SCIENTIFIC STATISTICAL ANALYSIS (MODULE-WISE)\n")
        f.write("=" * 100 + "\n\n")

        # Group by module for better readability
        for module in sorted(statistical_df['Module'].unique()):
            module_data = statistical_df[statistical_df['Module'] == module]
            f.write(f"Module: {module}\n")
            f.write("-" * 50 + "\n")

            for _, row in module_data.iterrows():
                f.write(f"Metric: {row['Metric']}\n")
                f.write(f"  Standard: {row['Standard Mean']} ± {row['Standard Std']}\n")
                f.write(f"  Semantic: {row['Semantic Mean']} ± {row['Semantic Std']}\n")
                f.write(f"  t-statistic: {row['t-statistic']}, p-value: {row['p-value']}\n")
                f.write(f"  Significant Difference: {row['Significant']}\n\n")

            f.write("\n")

        f.write("=" * 100 + "\n")
        f.write("Legend:\n")
        f.write("Standard/Semantic: Mean ± Standard Deviation\n")
        f.write("t-statistic: Welch's t-test statistic (unequal variances)\n")
        f.write("p-value: Two-tailed p-value (α = 0.05)\n")
        f.write("Significant: YES if p < 0.05, NO otherwise\n")

    return statistical_df


def _generate_improvement_metrics_table(df, available_metrics, output_path):
    """Generate additional improvement metrics table"""
    improvement_data = []

    for module in sorted(df['module'].unique()):
        module_df = df[df['module'] == module]

        std_df = module_df[module_df['semantics'] == False]
        sem_df = module_df[module_df['semantics'] == True]

        if std_df.empty or sem_df.empty:
            continue

        row = {'Module': module}

        for metric_key, metric_name in available_metrics.items():
            std_values = std_df[metric_key].dropna()
            sem_values = sem_df[metric_key].dropna()

            if len(std_values) > 0 and len(sem_values) > 0:
                # Coefficient of variation (stability measure)
                std_cv = std_values.std() / std_values.mean() if std_values.mean() > 0 else 0
                sem_cv = sem_values.std() / sem_values.mean() if sem_values.mean() > 0 else 0

                # Improvement in stability
                stability_improvement = std_cv - sem_cv

                # Success rate (percentage of runs above median of standard)
                std_median = std_values.median()
                success_rate = (sem_values > std_median).mean() * 100

                # Probability of superiority (area under curve)
                prob_superiority = np.mean([sem_val > std_val
                                            for sem_val in sem_values
                                            for std_val in std_values])

                row[f'{metric_name} Stability Improvement'] = f"{stability_improvement:.3f}"
                row[f'{metric_name} Success Rate (%)'] = f"{success_rate:.1f}%"
                row[f'{metric_name} Prob. Superiority'] = f"{prob_superiority:.3f}"
            else:
                row[f'{metric_name} Stability Improvement'] = "N/A"
                row[f'{metric_name} Success Rate (%)'] = "N/A"
                row[f'{metric_name} Prob. Superiority'] = "N/A"

        improvement_data.append(row)

    improvement_df = pd.DataFrame(improvement_data)

    # Save improvement metrics
    csv_path = os.path.join(output_path, 'improvement_metrics_table.csv')
    improvement_df.to_csv(csv_path, index=False)

    txt_path = os.path.join(output_path, 'improvement_metrics_table.txt')
    with open(txt_path, 'w') as f:
        f.write("IMPROVEMENT METRICS TABLE\n")
        f.write("=" * 100 + "\n\n")
        f.write(improvement_df.to_string(index=False))
        f.write("\n\n")
        f.write("Legend:\n")
        f.write("Stability Improvement: Reduction in coefficient of variation\n")
        f.write("Success Rate: % of semantic runs above standard median\n")
        f.write("Prob. Superiority: Probability semantic > standard (0.5 = no difference)\n")

    return improvement_df

def _plot_overall_timelines(timeline_df, vis_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metric_map = {
        'Coverage': 'coveragetimeline',
        'Test Suite Size': 'sizetimeline',
        'Test Suite Length': 'lengthtimeline',
        'Fitness': 'fitnesstimeline'
    }

    for idx, (metric_name, base_col) in enumerate(metric_map.items()):
        if idx >= 4:
            continue

        cols = [col for col in timeline_df.columns if col.startswith(base_col)]
        cols = sorted(cols, key=lambda x: int(x.split('_t')[-1]))

        if not cols:
            continue

        ax = axes[idx]
        x = range(1, len(cols) + 1)

        for semantics, color in zip([False, True], ['blue', 'orange']):
            subset = timeline_df[timeline_df['semantics'] == semantics]
            if subset.empty:
                continue

            vals = subset[cols].to_numpy(dtype=float)
            means = np.nanmean(vals, axis=0)
            stds = np.nanstd(vals, axis=0)

            ax.plot(x, means, label=f"{'Semantic' if semantics else 'Standard'}", color=color, linewidth=2)
            ax.fill_between(x, means - stds, means + stds, alpha=0.2, color=color)

            # Clean up arrays
            del vals, means, stds

        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Timeline - Overall')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'overall_timeline_comparison.png'), dpi=300)
    plt.close()

    del fig, axes
    gc.collect()


def _plot_convergence_analysis(timeline_df, vis_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metric_map = {
        'Coverage': 'coveragetimeline',
        'Test Suite Size': 'sizetimeline',
        'Test Suite Length': 'lengthtimeline',
        'Fitness': 'fitnesstimeline'
    }

    for idx, (metric_name, base_col) in enumerate(metric_map.items()):
        if idx >= 4:
            continue

        cols = [col for col in timeline_df.columns if col.startswith(base_col)]
        cols = sorted(cols, key=lambda x: int(x.split('_t')[-1]))

        if not cols:
            continue

        ax = axes[idx]
        convergence_data = []

        for semantics in [False, True]:
            subset = timeline_df[timeline_df['semantics'] == semantics]
            if subset.empty:
                continue

            # Process in smaller batches to reduce memory usage
            vals = subset[cols].to_numpy(dtype=float)

            for row in vals:
                if np.isnan(row[-1]):
                    continue

                final_val = row[-1]
                target = 0.9 * final_val if final_val > 0 else 0

                for i, val in enumerate(row):
                    if val >= target:
                        convergence_data.append({'semantics': semantics, 'convergence_iter': i + 1})
                        break

            # Clean up immediately
            del vals

        conv_df = pd.DataFrame(convergence_data)
        if not conv_df.empty:
            sns.boxplot(data=conv_df, x='semantics', y='convergence_iter', ax=ax)
            ax.set_title(f'{metric_name} - Convergence Speed (90%)')
            ax.set_xlabel('Semantic Enhancement')
            ax.set_ylabel('Iterations to Convergence')

        del conv_df, convergence_data

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'convergence_analysis.png'), dpi=300)
    plt.close()

    del fig, axes
    gc.collect()


def _plot_efficiency_boxplot(df, output_path):
    if 'searchtime' in df.columns:
        # Create a copy only for this specific calculation
        df_copy = df[['semantics', 'coverage', 'searchtime']].copy()
        df_copy['efficiency'] = df_copy['coverage'] / df_copy['searchtime']

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_copy, x='semantics', y='efficiency')
        plt.title('Coverage Efficiency (Coverage/SearchTime)')
        plt.savefig(os.path.join(output_path, 'efficiency_analysis.png'), dpi=300)
        plt.close()

        del df_copy

def _plot_test_suite_quality(df, output_path):
    quality_metrics = ['size', 'length']
    existing_metrics = [m for m in quality_metrics if m in df.columns]

    if not existing_metrics:
        return

    # Per-module plots
    modules = df['module'].unique() if 'module' in df.columns else []
    for module in modules:
        module_df = df[df['module'] == module]
        fig, axes = plt.subplots(1, len(existing_metrics), figsize=(15, 5))
        if len(existing_metrics) == 1:
            axes = [axes]
        for i, metric in enumerate(existing_metrics):
            sns.boxplot(data=module_df, x='semantics', y=metric, ax=axes[i])
            axes[i].set_title(f'{module} - Test Suite {metric.title()}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'test_suite_quality_{module}.png'), dpi=300)
        plt.close(fig)
        del fig, axes, module_df

    # Combined plot (all modules)
    fig, axes = plt.subplots(1, len(existing_metrics), figsize=(15, 5))
    if len(existing_metrics) == 1:
        axes = [axes]
    for i, metric in enumerate(existing_metrics):
        sns.boxplot(data=df, x='semantics', y=metric, ax=axes[i])
        axes[i].set_title(f'Test Suite {metric.title()} (All Modules)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'test_suite_quality.png'), dpi=300)
    plt.close(fig)
    del fig, axes

def run_statistical_analysis(run_results, output_base_path):
    # Create DataFrame only with needed columns
    df = pd.DataFrame([
        {**r.coverage_metrics, "module": r.module, "run": r.run_id, "semantics": r.semantics_enabled,
         "execution_time": r.execution_time}
        for r in run_results
    ])

    metrics = ['coverage', 'finalbranchcoverage', 'size', 'length', 'fitness']
    stat_results = []
    summary_file = os.path.join(output_base_path, "statistical_analysis.txt")

    with open(summary_file, 'w') as f:
        f.write("SCIENTIFIC STATISTICAL ANALYSIS\n" + "=" * 50 + "\n")
        for metric in metrics:
            if metric not in df.columns:
                continue

            std_vals = df[df['semantics'] == False][metric].dropna()
            sem_vals = df[df['semantics'] == True][metric].dropna()

            if len(std_vals) < 2 or len(sem_vals) < 2:
                continue

            tstat, pval = stats.ttest_ind(std_vals, sem_vals, equal_var=False)
            std_mean, std_std = std_vals.mean(), std_vals.std()
            sem_mean, sem_std = sem_vals.mean(), sem_vals.std()
            significant = pval < 0.05

            stat_result = StatisticalResults(
                metric_name=metric,
                standard_mean=std_mean,
                standard_std=std_std,
                semantic_mean=sem_mean,
                semantic_std=sem_std,
                p_value=pval,
                significant=significant
            )
            stat_results.append(stat_result)

            f.write(f"\nMetric: {metric}\n")
            f.write(f"  Standard: {std_mean:.4f} ± {std_std:.4f}\n")
            f.write(f"  Semantic: {sem_mean:.4f} ± {sem_std:.4f}\n")
            f.write(f"  t-statistic: {tstat:.4f}, p-value: {pval:.4f}\n")
            f.write(f"  Significant Difference: {'YES' if significant else 'NO'}\n")

        f.write("\n" + "=" * 50 + "\n")

    print(f"Statistical analysis summary written to {summary_file}")

def _plot_module_metric_barplot(df, metric, ylabel, title, filename, output_path):
    """
    Plot a bar chart for each module, showing mean metric value for standard and semantics,
    with error bars and overlay of individual runs (min, max, etc.).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plot_df = df[["module", "semantics", metric, "run"]].copy()
    plot_df = plot_df.dropna(subset=[metric, "module", "semantics", "run"])
    plot_df["module"] = plot_df["module"].astype(str)
    modules = sorted(plot_df["module"].unique())

    agg_stats = (
        plot_df
        .groupby(["module", "semantics"])[metric]
        .agg(["mean", "min", "max", "std", "median"])
        .reset_index()
    )
    print(f"Aggregated stats for {metric}:\n{agg_stats}")

    means = agg_stats.pivot(index="module", columns="semantics", values="mean").reindex(modules)
    stds = agg_stats.pivot(index="module", columns="semantics", values="std").reindex(modules)

    fig, ax = plt.subplots(figsize=(max(10, len(modules)*1.3), 7))

    bar_width = 0.35
    x = np.arange(len(modules))
    colors = ["#1f77b4", "#ff7f0e"]  # blue, orange

    rng = np.random.default_rng(42)  # for reproducibility

    for i, semantics in enumerate([False, True]):
        means_col = means[semantics]
        stds_col = stds[semantics]
        label = "Standard" if semantics is False else "Semantic"
        bar_pos = x - bar_width/2 + i*bar_width
        ax.bar(
            bar_pos,
            means_col,
            bar_width,
            yerr=stds_col,
            label=label,
            color=colors[i],
            capsize=5,
            alpha=0.85,
            edgecolor="black"
        )

        for j, mod in enumerate(modules):
            points = plot_df[(plot_df["module"] == mod) & (plot_df["semantics"] == semantics)][metric].values
            bar_x = bar_pos[j]
            if len(points) > 0:
                # Add jitter within the bar width
                jitter = rng.uniform(-bar_width/4, bar_width/4, size=len(points))
                scatter_x = np.full_like(points, bar_x, dtype=float) + jitter
                ax.scatter(
                    scatter_x,
                    points,
                    color=colors[i],
                    edgecolor="black",
                    alpha=0.7,
                    s=40,
                    zorder=10,
                    linewidth=0.7
                )
                # Draw single vertical line from min to max
                min_val = np.min(points)
                max_val = np.max(points)
                ax.plot(
                    [bar_x, bar_x],
                    [min_val, max_val],
                    color='black',
                    linewidth=2,
                    zorder=6
                )

    ax.set_xlabel("Module")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(modules, rotation=30, ha="right")
    ax.legend(title="Mode", loc="best")
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename), dpi=300)
    plt.close(fig)
