"""
MedTriage-Env: FastAPI server exposing OpenEnv HTTP API.
Endpoints: GET /reset, POST /step, GET /state, GET /health, GET /tasks
"""

from __future__ import annotations
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import TriageAction, TriageObservation, EnvironmentState, StepResult
from environment import MedTriageEnv
from tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MedTriage-Env",
    description=(
        "Clinical note triage OpenEnv environment. "
        "An AI agent learns to triage patients — assigning urgency levels, "
        "routing to departments, and identifying missing critical information."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instances per task (stateful, single-user for demo)
_envs: Dict[str, MedTriageEnv] = {}

VALID_TASKS = list(TASK_REGISTRY.keys())


def _get_env(task_id: str) -> MedTriageEnv:
    if task_id not in VALID_TASKS:
        raise HTTPException(400, f"Invalid task_id. Choose from: {VALID_TASKS}")
    if task_id not in _envs:
        _envs[task_id] = MedTriageEnv(task_id=task_id)
    return _envs[task_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """Health check — must return 200."""
    return {"status": "ok", "env": "MedTriage-Env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "task_id": tid,
                "name": cfg["name"],
                "difficulty": cfg["difficulty"],
                "description": cfg["description"],
                "max_steps": cfg["max_steps"],
            }
            for tid, cfg in TASK_REGISTRY.items()
        ]
    }


@app.post("/reset")
@app.get("/reset")
def reset(task_id: str = "task1_single") -> TriageObservation:
    """
    Reset the environment for a given task.
    Returns the initial observation.
    """
    env = _get_env(task_id)
    obs = env.reset()
    return obs


@app.post("/step")
def step(action: TriageAction, task_id: str = "task1_single") -> StepResult:
    """
    Submit a triage action and receive the next observation + reward.
    """
    env = _get_env(task_id)
    result = env.step(action)
    return result


@app.get("/state")
def state(task_id: str = "task1_single") -> EnvironmentState:
    """Return current environment state."""
    env = _get_env(task_id)
    return env.state()


@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint with environment overview."""
    return {
        "name": "MedTriage-Env",
        "description": "Clinical note triage OpenEnv environment",
        "version": "1.0.0",
        "tasks": VALID_TASKS,
        "endpoints": {
            "health": "GET /health",
            "reset": "POST /reset?task_id=<task_id>",
            "step": "POST /step?task_id=<task_id>",
            "state": "GET /state?task_id=<task_id>",
            "tasks": "GET /tasks",
            "docs": "GET /docs",
        },
    }


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)