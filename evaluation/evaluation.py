# evaluation/evaluation.py
import os
from datetime import datetime

import numpy as np
import pandas as pd
from .config import ExperimentConfig, PynguinRunConfig
from .runner import PynguinRunner
from .analysis import AnalysisPlots, run_statistical_analysis

class PynguinEvaluation:
    def __init__(self, config: ExperimentConfig, run_config: PynguinRunConfig):
        self.config = config
        self.run_config = run_config
        self.run_results = []

    def evaluate(self):
        for module_name in self.config.modules:
            print(f"\n--- Testing module: {module_name} ---")
            self._run_for_mode(module_name, run_semantics=False)
            self._run_for_mode(module_name, run_semantics=True)
            self._save_module_summary(module_name)
        if self.run_results:
            self._save_all_results()
            AnalysisPlots(self.config.output_base_path, self.run_config).full_analysis(self.run_results, self.config.modules)
            run_statistical_analysis(self.run_results, self.config.output_base_path)
        else:
            print("No results were collected!")

    def _run_for_mode(self, module_name, run_semantics):
        runner = PynguinRunner(self.config)
        label = "Semantic" if run_semantics else "Standard"
        for run in range(1, self.config.num_runs + 1):
            print(f"{label} run {run}/{self.config.num_runs}")
            result = runner.run(module_name, run, use_semantics=run_semantics)
            if result:
                self.run_results.append(result)
            else:
                print(f"  Pynguin run failed for {module_name} ({label} run {run})")

    def _save_module_summary(self, module_name):
        module_results = [r for r in self.run_results if r.module == module_name]
        if not module_results:
            return
        module_df = pd.DataFrame([
            {**r.coverage_metrics, "run": r.run_id, "semantics": r.semantics_enabled} for r in module_results
        ])
        summary_file = os.path.join(self.config.output_base_path, f"{module_name}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(self._generate_module_summary_text(module_df, module_name, self.config.num_runs))
        print(f"Module summary for {module_name} written to {summary_file}")

    def _generate_module_summary_text(self, module_df, module_name, num_runs):
        """Generate a textual summary for a specific module."""
        lines = []
        lines.append(f"MODULE EVALUATION SUMMARY: {module_name}\n{'=' * 50}\n")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nNumber of runs: {num_runs}\n\n")
        for semantics, label in zip([False, True], ["Standard", "Semantic"]):
            subset = module_df[module_df['semantics'] == semantics]
            if subset.empty:
                continue
            lines.append(f"{label} Mode:\n")
            for col in subset.select_dtypes(include=[np.number]).columns:
                if col not in ['run', 'semantics']:
                    mean, std = subset[col].mean(), subset[col].std()
                    lines.append(f"  {col}: {mean:.4f} ± {std:.4f}\n")
            lines.append('\n')
        lines.append('=' * 50 + '\n')
        return ''.join(lines)

    def _save_all_results(self):
        results_df = pd.DataFrame([
            {**r.coverage_metrics, "module": r.module, "run": r.run_id, "semantics": r.semantics_enabled,
             "execution_time": r.execution_time}
            for r in self.run_results
        ])
        results_file = os.path.join(self.config.output_base_path, "coverage_comparison.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")
