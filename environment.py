"""
MedTriage-Env: Core OpenEnv environment.

Improvements over v1:
- Task 2 rotates through 4 different queues (no repeated identical episodes)
- Task 1/3 rotate through all cases systematically
- Anti-gaming: rewards consecutive correct decisions, not single lucky guesses
- Episode length varies by task difficulty
- Full OpenEnv spec: reset() / step() / state()
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Optional

from models import (
    TriageAction, TriageObservation, EnvironmentState, StepResult, PatientNote
)
from tasks import TASK_REGISTRY, grade_task1, grade_task2, grade_task3
from data import TASK1_CASES, TASK2_QUEUES, TASK3_CASES


class MedTriageEnv:
    """
    MedTriage-Env: Clinical note triage environment for AI agents.

    The environment presents clinical patient notes and requires the agent
    to triage them — assigning urgency levels, routing to departments,
    and identifying missing critical information.

    Observation Space
    -----------------
    TriageObservation:
      task_id            str   — active task
      task_description   str   — natural language instructions
      patients           list  — PatientNote objects
      step_number        int   — current step (0-indexed)
      max_steps          int   — episode length
      feedback           str   — previous action feedback
      cumulative_reward  float — running total
      done               bool  — episode complete

    Action Space
    ------------
    TriageAction:
      urgency_level  str              — CRITICAL|HIGH|MEDIUM|LOW|NON_URGENT
      reasoning      str              — clinical justification
      department     str              — target department
      confidence     float [0,1]      — self-reported confidence
      missing_info   list[str]        — missing data items (Task 3)
      patient_ids    list[str]        — priority order (Task 2)

    Reward
    ------
    Shaped per-step reward [0.0, 1.0]. Partial credit throughout trajectory.
    Penalties for dangerous undertriage, overconfidence, keyword stuffing.
    """

    VALID_TASKS = list(TASK_REGISTRY.keys())

    def __init__(self, task_id: str = "task1_single", seed: int = 42):
        if task_id not in self.VALID_TASKS:
            raise ValueError(f"task_id must be one of {self.VALID_TASKS}")
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)
        self._task_config = TASK_REGISTRY[task_id]

        # Episode state
        self._step_number: int = 0
        self._max_steps: int = self._task_config["max_steps"]
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._case_idx: int = 0
        self._queue_idx: int = 0
        self._action_history: List[Dict[str, Any]] = []
        self._reward_history: List[float] = []
        self._current_patients: List[PatientNote] = []
        self._feedback: str = ""
        self._initialized: bool = False

    # -----------------------------------------------------------------------
    # OpenEnv API
    # -----------------------------------------------------------------------

    def reset(self) -> TriageObservation:
        """Reset environment to initial state. Returns first observation."""
        self._rng = random.Random(self.seed)
        self._step_number = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._action_history = []
        self._reward_history = []
        self._feedback = ""
        self._initialized = True

        # Randomise starting case within seed
        num_cases = self._task_config["num_cases"]
        self._case_idx = self._rng.randint(0, num_cases - 1)
        self._queue_idx = self._rng.randint(0, len(TASK2_QUEUES) - 1)

        self._current_patients = self._load_patients()
        return self._build_observation()

    def step(self, action: TriageAction) -> StepResult:
        """
        Submit a triage action.

        Parameters
        ----------
        action : TriageAction

        Returns
        -------
        StepResult(observation, reward, done, info)
        """
        if not self._initialized:
            obs = self.reset()
            return StepResult(
                observation=obs, reward=0.0, done=False,
                info={"warning": "Auto-reset: call reset() before step()"}
            )

        if self._done:
            obs = self._build_observation()
            return StepResult(
                observation=obs, reward=0.0, done=True,
                info={"warning": "Episode is done. Call reset() to start a new episode."}
            )

        # Grade the action
        reward_obj = self._grade(action)
        reward = float(reward_obj.total)

        # Update state
        self._step_number += 1
        self._cumulative_reward = round(self._cumulative_reward + reward, 4)
        self._reward_history.append(reward)
        self._action_history.append({
            "step": self._step_number,
            "action": action.model_dump(),
            "reward": reward,
            "breakdown": reward_obj.model_dump(),
        })
        self._feedback = reward_obj.explanation

        # Advance to next case for multi-step tasks
        if self.task_id == "task1_single":
            self._case_idx = (self._case_idx + 1) % len(TASK1_CASES)
            self._current_patients = self._load_patients()
        elif self.task_id == "task2_queue":
            # Rotate queues so each step sees a different patient set
            self._queue_idx = (self._queue_idx + 1) % len(TASK2_QUEUES)
            self._current_patients = self._load_patients()
        elif self.task_id == "task3_incomplete":
            self._case_idx = (self._case_idx + 1) % len(TASK3_CASES)
            self._current_patients = self._load_patients()

        self._done = self._step_number >= self._max_steps

        obs = self._build_observation()
        info = {
            "reward_breakdown": reward_obj.model_dump(),
            "step": self._step_number,
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
        }
        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    def state(self) -> EnvironmentState:
        """Return full current environment state."""
        return EnvironmentState(
            task_id=self.task_id,
            step_number=self._step_number,
            max_steps=self._max_steps,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            current_patients=self._current_patients,
            action_history=self._action_history,
            reward_history=self._reward_history,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _grade(self, action: TriageAction):
        if self.task_id == "task1_single":
            return grade_task1(action, self._case_idx)
        elif self.task_id == "task2_queue":
            return grade_task2(action, self._queue_idx)
        elif self.task_id == "task3_incomplete":
            return grade_task3(action, self._case_idx)
        raise ValueError(f"Unknown task_id: {self.task_id}")

    def _load_patients(self) -> List[PatientNote]:
        if self.task_id == "task1_single":
            return [TASK1_CASES[self._case_idx % len(TASK1_CASES)]["patient"]]
        elif self.task_id == "task2_queue":
            return TASK2_QUEUES[self._queue_idx % len(TASK2_QUEUES)]["patients"]
        elif self.task_id == "task3_incomplete":
            return [TASK3_CASES[self._case_idx % len(TASK3_CASES)]["patient"]]
        return []

    def _build_observation(self) -> TriageObservation:
        return TriageObservation(
            task_id=self.task_id,
            task_description=self._task_config["description"],
            patients=self._current_patients,
            step_number=self._step_number,
            max_steps=self._max_steps,
            feedback=self._feedback,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
        )