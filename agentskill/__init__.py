"""agentskill — learning general agent skills from public trajectories.

Collect agent trajectories -> score their quality -> retrieve relevant past
experiences -> learn (retrieval-augmented imitation now; LoRA distillation on
GPU) -> evaluate baseline vs trajectory-learned across GAIA / MLE-bench /
SWE-bench-style suites. Offline and deterministic end to end; the GPU steps
are opt-in for Colab.
"""

from .agents import BaselineAgent, TrajectoryLearnedAgent
from .benchmark import Task, grade, mock_suite
from .evaluate import compare, format_report
from .memory import TrajectoryMemory
from .scoring import curate, rank, score_trajectory
from .trajectory import Trajectory, synth_dataset

__all__ = [
    "BaselineAgent", "Task", "Trajectory", "TrajectoryLearnedAgent",
    "TrajectoryMemory", "compare", "curate", "format_report", "grade",
    "mock_suite", "rank", "score_trajectory", "synth_dataset",
]
