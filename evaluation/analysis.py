# evaluation/analysis.py
import os
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
        timeline_results = [
            {**r.timeline_data, "module": r.module, "run": r.run_id, "semantics": r.semantics_enabled}
            for r in run_results if r.timeline_data
        ]
        self.visualize_timelines(timeline_results)
        all_results = [
            {**r.coverage_metrics, "module": r.module, "run": r.run_id, "semantics": r.semantics_enabled,
             "execution_time": r.execution_time}
            for r in run_results
        ]
        self.comprehensive_stats(all_results, modules)
        self.correlation_heatmap(all_results)

    def visualize_timelines(self, timeline_results):
        if not timeline_results:
            print("No timeline data available for visualization.")
            return
        timeline_df = pd.DataFrame(timeline_results)
        vis_dir = os.path.join(self.output_path, 'timeline_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        for module in timeline_df['module'].unique():
            _plot_module_timelines(timeline_df[timeline_df['module'] == module], module, vis_dir, self.run_config)
        _plot_overall_timelines(timeline_df, vis_dir)
        _plot_convergence_analysis(timeline_df, vis_dir)
        print(f"Timeline visualizations saved to {vis_dir}")

    def comprehensive_stats(self, all_results, modules):
        if not all_results:
            print("No final results to analyze.")
            return
        df = pd.DataFrame(all_results)
        _plot_coverage_vs_complexity(df, self.output_path)
        _plot_efficiency_boxplot(df, self.output_path)
        _plot_test_suite_quality(df, self.output_path)
        _plot_metric_violinplots(df, self.output_path)
        _plot_metric_boxplots_with_significance(df, self.output_path)

    def correlation_heatmap(self, all_results):
        df = pd.DataFrame(all_results)
        numeric = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 8))
        corr = numeric.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap of Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "correlation_heatmap.png"), dpi=300)
        plt.close()

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
    interval_sec = pynguin_config.timeline_interval / 1_000_000_000  # convert ns to s
    for idx, (metric_name, base_col) in enumerate(metric_map.items()):
        cols = [col for col in module_data.columns if col.startswith(base_col)]
        cols = sorted(cols, key=lambda x: int(x.split('_t')[-1]))
        if not cols or idx >= 4: continue
        ax = axes[idx]
        x = [interval_sec * i for i in range(len(cols))]  # time in seconds
        for semantics, color in zip([False, True], ['blue', 'orange']):
            subset = module_data[module_data['semantics'] == semantics]
            if subset.empty: continue
            vals = subset[cols].to_numpy(dtype=float)
            means = np.nanmean(vals, axis=0)
            stds = np.nanstd(vals, axis=0)
            label = f"{'Semantic' if semantics else 'Standard'} (n={len(subset)})"
            ax.plot(x, means, label=label, color=color, linewidth=2)
            ax.fill_between(x, means - stds, means + stds, alpha=0.2, color=color)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Timeline - {module_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{module_name}_timeline_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

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
        cols = [col for col in timeline_df.columns if col.startswith(base_col)]
        cols = sorted(cols, key=lambda x: int(x.split('_t')[-1]))
        if not cols or idx >= 4: continue
        ax = axes[idx]
        x = range(1, len(cols) + 1)
        for semantics, color in zip([False, True], ['blue', 'orange']):
            subset = timeline_df[timeline_df['semantics'] == semantics]
            if subset.empty: continue
            vals = subset[cols].to_numpy(dtype=float)
            means = np.nanmean(vals, axis=0)
            stds = np.nanstd(vals, axis=0)
            ax.plot(x, means, label=f"{'Semantic' if semantics else 'Standard'}", color=color, linewidth=2)
            ax.fill_between(x, means - stds, means + stds, alpha=0.2, color=color)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Timeline - Overall')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'overall_timeline_comparison.png'), dpi=300)
    plt.close()

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
        cols = [col for col in timeline_df.columns if col.startswith(base_col)]
        cols = sorted(cols, key=lambda x: int(x.split('_t')[-1]))
        if not cols or idx >= 4: continue
        ax = axes[idx]
        convergence_data = []
        for semantics in [False, True]:
            subset = timeline_df[timeline_df['semantics'] == semantics]
            if subset.empty: continue
            vals = subset[cols].to_numpy(dtype=float)
            for row in vals:
                if np.isnan(row[-1]):
                    continue
                final_val = row[-1]
                target = 0.9 * final_val if final_val > 0 else 0
                for i, val in enumerate(row):
                    if val >= target:
                        convergence_data.append({'semantics': semantics, 'convergence_iter': i+1})
                        break
        conv_df = pd.DataFrame(convergence_data)
        if not conv_df.empty:
            sns.boxplot(data=conv_df, x='semantics', y='convergence_iter', ax=ax)
            ax.set_title(f'{metric_name} - Convergence Speed (90%)')
            ax.set_xlabel('Semantic Enhancement')
            ax.set_ylabel('Iterations to Convergence')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'convergence_analysis.png'), dpi=300)
    plt.close()

def _plot_coverage_vs_complexity(df, output_path):
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
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Coverage vs. Complexity Analysis', fontsize=16)
    plt.savefig(os.path.join(output_path, 'coverage_vs_complexity.png'), dpi=300)
    plt.close()

def _plot_efficiency_boxplot(df, output_path):
    if 'searchtime' in df.columns:
        df = df.copy()
        df['efficiency'] = df['coverage'] / df['searchtime']
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='semantics', y='efficiency')
        plt.title('Coverage Efficiency (Coverage/SearchTime)')
        plt.savefig(os.path.join(output_path, 'efficiency_analysis.png'), dpi=300)
        plt.close()

def _plot_test_suite_quality(df, output_path):
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

def run_statistical_analysis(run_results, output_base_path):
    df = pd.DataFrame([
        {**r.coverage_metrics, "module": r.module, "run": r.run_id, "semantics": r.semantics_enabled,
         "execution_time": r.execution_time}
        for r in run_results
    ])
    metrics = ['coverage', 'finalbranchcoverage', 'size', 'length', 'fitness']
    stat_results = []
    summary_file = os.path.join(output_base_path, "statistical_analysis.txt")
    with open(summary_file, 'w') as f:
        f.write("SCIENTIFIC STATISTICAL ANALYSIS\n" + "="*50 + "\n")
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
        f.write("\n" + "="*50 + "\n")
    print(f"Statistical analysis summary written to {summary_file}")



def _plot_metric_violinplots(df, output_path):
    metrics = ["coverage", "finalbranchcoverage", "size", "length", "fitness"]
    for metric in metrics:
        if metric in df.columns:
            plt.figure(figsize=(8, 6))
            sns.violinplot(data=df, x="semantics", y=metric, hue="semantics", inner="box", palette="Set2", legend=False)
            plt.title(f"Distribution of {metric.title()} by Semantics")
            plt.savefig(os.path.join(output_path, f"{metric}_violinplot.png"), dpi=300)
            plt.close()

def _plot_metric_boxplots_with_significance(df, output_path):
    metrics = ["coverage", "finalbranchcoverage", "size", "length", "fitness"]
    for metric in metrics:
        if metric in df.columns:
            plt.figure(figsize=(8, 6))
            ax = sns.boxplot(data=df, x="semantics", y=metric, hue="semantics", showfliers=True, palette="Set1",
                             legend=False)
            # Statistical annotation
            std_vals = df[df["semantics"] == False][metric].dropna()
            sem_vals = df[df["semantics"] == True][metric].dropna()
            if len(std_vals) > 1 and len(sem_vals) > 1:
                tstat, pval = stats.ttest_ind(std_vals, sem_vals, equal_var=False)
                sig = ""
                if pval < 0.001:
                    sig = "***"
                elif pval < 0.01:
                    sig = "**"
                elif pval < 0.05:
                    sig = "*"
                ax.text(0.5, max(df[metric]), f"p={pval:.3g} {sig}", ha="center", va="bottom", fontsize=12)
            plt.title(f"{metric.title()} by Semantics (Boxplot with Outliers)")
            plt.savefig(os.path.join(output_path, f"{metric}_boxplot_significance.png"), dpi=300)
            plt.close()
