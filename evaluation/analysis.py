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
                    "semantics": r.semantics_enabled
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
        self.module_barplots(all_results)  # <--- NEW: generate module-wise barplots

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

    interval_sec = pynguin_config.timeline_interval / 1_000_000_000

    for idx, (metric_name, base_col) in enumerate(metric_map.items()):
        if idx >= 4:
            continue

        cols = [col for col in module_data.columns if col.startswith(base_col)]
        cols = sorted(cols, key=lambda x: int(x.split('_t')[-1]))

        if not cols:
            continue

        ax = axes[idx]
        x = [interval_sec * i for i in range(len(cols))]

        colors = ['#2E86AB', '#A23B72']
        linestyles = ['-', '--']
        alphas = [0.8, 0.8]
        fill_alphas = [0.15, 0.15]

        for i, (semantics, color, linestyle, alpha, fill_alpha) in enumerate(
            zip([False, True], colors, linestyles, alphas, fill_alphas)):
            subset = module_data[module_data['semantics'] == semantics]
            if subset.empty:
                continue

            # Process data in smaller chunks to reduce memory usage
            vals = subset[cols].to_numpy(dtype=float)
            means = np.nanmean(vals, axis=0)
            stds = np.nanstd(vals, axis=0)

            label = f"{'Semantic' if semantics else 'Standard'} (n={len(subset)})"

            # Plot line with transparency and different styles
            ax.plot(x, means, label=label, color=color, linewidth=2.5,
                    linestyle=linestyle, alpha=alpha, zorder=3 - i)

            # Add fill with lower transparency
            ax.fill_between(x, means - stds, means + stds, alpha=fill_alpha,
                            color=color, zorder=1 - i)

            # Clean up arrays immediately
            del vals, means, stds

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Timeline - {module_name}')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{module_name}_timeline_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Force cleanup
    del fig, axes
    gc.collect()


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
