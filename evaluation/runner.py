# evaluation/runner.py
import os
import hashlib
import time
import logging
import dataclasses
from .config import ExperimentConfig, RunResults, PynguinRunConfig
from .data import extract_coverage_data

class PynguinRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self, module_name: str, run_iteration: int, use_semantics: bool):
        from pynguin.configuration import (
            Configuration, TestCaseOutputConfiguration, StatisticsOutputConfiguration,
            StoppingConfiguration, SearchAlgorithmConfiguration, SeedingConfiguration, Algorithm,
            CoverageMetric, Minimization, MinimizationStrategy
        )
        from pynguin.utils.statistics.runtimevariable import RuntimeVariable
        from pynguin import set_configuration, run_pynguin as execute_pynguin

        run_dir = os.path.join(self.config.output_runs_path, f"{module_name}_{'semantic' if use_semantics else 'standard'}_run{run_iteration}")
        report_dir = os.path.join(run_dir, "pynguin-report")
        os.makedirs(report_dir, exist_ok=True)
        os.environ["PYNGUIN_DANGER_AWARE"] = "1"
        os.environ["PYTHONHASHSEED"] = "0"

        seed = int.from_bytes(hashlib.sha256(str(run_iteration).encode()).digest()[:4], 'big')

        config = Configuration(
            project_path=self.config.project_path,
            module_name=module_name,
            test_case_output=TestCaseOutputConfiguration(
                output_path=run_dir,
                # assertion_generation="MUTATION_ANALYSIS",
                assertion_generation="CHECKED_MINIMIZING",
                minimization=Minimization(
                    test_case_minimization_strategy=MinimizationStrategy.NONE
                ),
            ),
            statistics_output=StatisticsOutputConfiguration(
                report_dir=report_dir,
                statistics_backend="CSV",
                coverage_metrics=[CoverageMetric("BRANCH")],
                timeline_interval=PynguinRunConfig.timeline_interval,
                timeline_interpolation=False,
                output_variables=[
                    RuntimeVariable.TargetModule, RuntimeVariable.TotalTime, RuntimeVariable.SearchTime,
                    RuntimeVariable.AlgorithmIterations, RuntimeVariable.Coverage, RuntimeVariable.BranchCoverage,
                    RuntimeVariable.FinalBranchCoverage, RuntimeVariable.Size, RuntimeVariable.Length,
                    RuntimeVariable.FinalSize, RuntimeVariable.FinalLength, RuntimeVariable.Fitness,
                    RuntimeVariable.CoverageTimeline, RuntimeVariable.SizeTimeline,
                    RuntimeVariable.LengthTimeline, RuntimeVariable.FitnessTimeline,
                    RuntimeVariable.Goals, RuntimeVariable.Predicates, RuntimeVariable.Lines,
                    RuntimeVariable.McCabeAST, RuntimeVariable.AccessibleObjectsUnderTest,
                ],
            ),
            stopping=StoppingConfiguration(
                maximum_iterations=self.config.max_iterations,
                maximum_search_time=self.config.max_search_time,
                maximum_memory=self.config.max_memory,
            ),
            search_algorithm=SearchAlgorithmConfiguration(population=self.config.population_size),
            seeding=SeedingConfiguration(seed=seed),
            algorithm=Algorithm.DYNAMOSA,
        )

        if use_semantics:
            config.enable_seed_examples = True
            config.use_docstring_semantics = True

        if self.config.verbose:
            logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
            logger = logging.getLogger("pynguin")
            logger.setLevel(logging.INFO)
            print(f"Pynguin config for {module_name} (semantics={use_semantics}):")
            for field in dataclasses.fields(config):
                print(f"  {field.name}: {getattr(config, field.name)}")

        try:
            set_configuration(config)
            start_time = time.time()
            execute_pynguin()
            exec_time = time.time() - start_time
            stats_file = os.path.join(report_dir, "statistics.csv")
            coverage_data, timeline_data = extract_coverage_data(stats_file)
            return RunResults(
                module=module_name,
                run_id=run_iteration,
                semantics_enabled=use_semantics,
                execution_time=exec_time,
                coverage_metrics=coverage_data,
                timeline_data=timeline_data.iloc[0].to_dict() if timeline_data is not None else {}
            )
        except Exception as e:
            print(f"[ERROR] Pynguin run failed: {e}")
            return None
