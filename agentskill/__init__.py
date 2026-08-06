"""agentskill — learning general agent skills from public trajectories.

Collect agent trajectories -> score their quality -> retrieve relevant past
experiences -> learn (retrieval-augmented imitation now; LoRA distillation on
GPU) -> evaluate baseline vs trajectory-learned across GAIA / MLE-bench /
SWE-bench-style suites. Offline and deterministic end to end; the GPU steps
are opt-in for Colab.
"""

from .agents import (BaselineAgent, BaselinePolicy, LearnedPolicy,
                     TrajectoryLearnedAgent, ablation_policies, consensus_plan)
from .benchmark import Task, grade, mock_suite
from .env import Episode, ToolEnv
from .evaluate import compare, format_full_report, format_report, full_evaluation
from .memory import TrajectoryMemory
from .mining import RecoveryRule, mine_recovery, mine_skills
from .scoring import curate, rank, score_trajectory
from .trajectory import Trajectory, synth_dataset

__all__ = [
    "BaselineAgent", "BaselinePolicy", "Episode", "LearnedPolicy",
    "RecoveryRule", "Task", "ToolEnv", "Trajectory", "TrajectoryLearnedAgent",
    "TrajectoryMemory", "ablation_policies", "compare", "consensus_plan",
    "curate", "format_full_report", "format_report", "full_evaluation",
    "grade", "mine_recovery", "mine_skills", "mock_suite", "rank",
    "score_trajectory", "synth_dataset",
]
