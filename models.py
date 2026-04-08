"""
MedTriage-Env: Typed Pydantic models for OpenEnv compliance.
Observation, Action, Reward, and State models.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    """
    The agent's response to a triage scenario.

    Fields
    ------
    urgency_level : str
        One of "CRITICAL", "HIGH", "MEDIUM", "LOW", "NON_URGENT"
    reasoning : str
        Free-text explanation of the triage decision (required).
    missing_info : List[str]
        List of missing data items the agent flags (used in Task 3).
    department : str
        Target department, e.g. "Emergency", "Cardiology", "General".
    confidence : float
        Agent self-reported confidence 0.0–1.0.
    patient_ids : List[str]
        For Task 2 (queue), ordered list of patient IDs by priority (high → low).
    """
    urgency_level: str = Field(
        default="MEDIUM",
        description="CRITICAL | HIGH | MEDIUM | LOW | NON_URGENT"
    )
    reasoning: str = Field(
        default="",
        description="Explanation for the triage decision"
    )
    missing_info: List[str] = Field(
        default_factory=list,
        description="List of missing critical data items identified"
    )
    department: str = Field(
        default="General",
        description="Target department for routing"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent confidence in its decision (0.0–1.0)"
    )
    patient_ids: List[str] = Field(
        default_factory=list,
        description="Ordered patient IDs by priority (Task 2 queue)"
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class PatientNote(BaseModel):
    """A single patient's clinical note presented to the agent."""
    patient_id: str
    chief_complaint: str
    vitals: Dict[str, Any]
    history: str
    symptoms: List[str]
    duration: str
    age: int
    sex: str
    notes: Optional[str] = None


class TriageObservation(BaseModel):
    """
    Observation returned after reset() or step().

    Fields
    ------
    task_id : str
        Which task is active: "task1_single", "task2_queue", "task3_incomplete"
    task_description : str
        Natural language description of what the agent must do.
    patients : List[PatientNote]
        One or more patient notes to triage.
    step_number : int
        Current step within the episode.
    max_steps : int
        Maximum steps allowed.
    feedback : str
        Feedback from the previous action (empty on first step).
    cumulative_reward : float
        Running reward total so far.
    done : bool
        Whether the episode is complete.
    """
    task_id: str
    task_description: str
    patients: List[PatientNote]
    step_number: int = 0
    max_steps: int = 5
    feedback: str = ""
    cumulative_reward: float = 0.0
    done: bool = False


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class TriageReward(BaseModel):
    """Structured reward breakdown."""
    total: float = Field(ge=0.0, le=1.0)
    urgency_score: float = Field(ge=0.0, le=1.0, default=0.0)
    department_score: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning_score: float = Field(ge=0.0, le=1.0, default=0.0)
    missing_info_score: float = Field(ge=0.0, le=1.0, default=0.0)
    ordering_score: float = Field(ge=0.0, le=1.0, default=0.0)
    penalty: float = Field(ge=0.0, le=1.0, default=0.0)
    explanation: str = ""


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: TriageObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    task_id: str
    step_number: int
    max_steps: int
    cumulative_reward: float
    done: bool
    current_patients: List[PatientNote]
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    reward_history: List[float] = Field(default_factory=list)