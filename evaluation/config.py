# evaluation/config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class ExperimentConfig:
    project_path: str
    output_base_path: str
    output_runs_path: str
    modules: List[str]
    num_runs: int = 5
    max_iterations: int = 20
    max_search_time: int = 300
    max_memory: int = 5000
    population_size: int = 50
    verbose: bool = False

@dataclass
class PynguinRunConfig:
    timeline_interval: int = 200_000_000 # nanoseconds

@dataclass
class RunResults:
    module: str
    run_id: int
    semantics_enabled: bool
    execution_time: float
    coverage_metrics: Dict[str, Any]
    timeline_data: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class StatisticalResults:
    metric_name: str
    standard_mean: float
    standard_std: float
    semantic_mean: float
    semantic_std: float
    p_value: float
    significant: bool
