"""
MedTriage-Env — Baseline Inference Script
==========================================
Uses OpenAI client to run a model against all 3 MedTriage tasks.
Emits structured [START], [STEP], [END] logs for automated evaluation.

Usage:
    export API_BASE_URL=<your_api_base>
    export MODEL_NAME=<model_identifier>
    export HF_TOKEN=<your_hf_token>
    python inference.py
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK: str = "MedTriage-Env"
MAX_STEPS: int = 5
MAX_TOTAL_REWARD: float = float(MAX_STEPS)
SUCCESS_SCORE_THRESHOLD: float = 0.6

TASKS = ["task1_single", "task2_queue", "task3_incomplete"]

# ---------------------------------------------------------------------------
# Structured logging (required format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({
        "type": "START",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(json.dumps({
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error,
        "timestamp": time.time(),
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({
        "type": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
        "timestamp": time.time(),
    }), flush=True)


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

class MedTriageHTTPEnv:
    """Simple HTTP client for the MedTriage-Env FastAPI server."""

    def __init__(self, base_url: str = ENV_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)

    def reset(self, task_id: str) -> Dict[str, Any]:
        r = self._client.post(f"{self.base_url}/reset", params={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, task_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self._client.post(
            f"{self.base_url}/step",
            params={"task_id": task_id},
            json=action,
        )
        r.raise_for_status()
        return r.json()

    def state(self, task_id: str) -> Dict[str, Any]:
        r = self._client.get(f"{self.base_url}/state", params={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Agent: LLM decision-making
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an experienced emergency medicine physician performing rapid patient triage.
Your goal is to assess patients and assign appropriate urgency levels and department routing.

Urgency levels (use EXACTLY these):
- CRITICAL: Immediate life-threatening, act within minutes
- HIGH: Urgent, act within 15-30 minutes  
- MEDIUM: Semi-urgent, act within 1-2 hours
- LOW: Non-urgent, can wait several hours
- NON_URGENT: Routine, no immediate urgency

For each patient or queue, respond with a JSON object containing:
{
  "urgency_level": "CRITICAL|HIGH|MEDIUM|LOW|NON_URGENT",
  "reasoning": "Your clinical reasoning (2-4 sentences citing specific findings)",
  "department": "Emergency|Cardiology|Pulmonology|Orthopedics|Neurology|Ophthalmology|General",
  "confidence": 0.0-1.0,
  "missing_info": ["list", "of", "missing", "critical", "data"],
  "patient_ids": ["ordered", "list", "by", "priority", "for", "queue", "tasks"]
}

ALWAYS respond with valid JSON only. No preamble."""


def format_patient_for_llm(patient: Dict[str, Any]) -> str:
    return (
        f"Patient ID: {patient['patient_id']}\n"
        f"Age/Sex: {patient['age']}y {patient['sex']}\n"
        f"Chief Complaint: {patient['chief_complaint']}\n"
        f"Vitals: {patient['vitals']}\n"
        f"Symptoms: {', '.join(patient['symptoms'])}\n"
        f"Duration: {patient['duration']}\n"
        f"History: {patient['history']}\n"
        f"Notes: {patient.get('notes', 'None')}"
    )


def build_user_prompt(obs: Dict[str, Any], history: List[str], last_reward: float) -> str:
    task_id = obs["task_id"]
    task_desc = obs["task_description"]
    patients = obs["patients"]
    feedback = obs.get("feedback", "")

    patient_text = "\n\n---\n\n".join(
        format_patient_for_llm(p) for p in patients
    )

    prompt = f"TASK: {task_desc}\n\n"
    prompt += f"PATIENT(S):\n{patient_text}\n\n"

    if feedback:
        prompt += f"Previous feedback: {feedback}\n"
        prompt += f"Previous reward: {last_reward:.3f}\n\n"

    if task_id == "task2_queue":
        ids = [p["patient_id"] for p in patients]
        prompt += f"Patient IDs to order: {ids}\nReturn patient_ids in priority order (highest priority first).\n\n"

    if task_id == "task3_incomplete":
        prompt += "IMPORTANT: Identify ALL missing critical information. List each item in 'missing_info'. Express appropriate uncertainty in confidence score.\n\n"

    prompt += "Respond with JSON only."
    return prompt


def get_model_action(
    client: OpenAI,
    obs: Dict[str, Any],
    history: List[str],
    last_reward: float,
) -> Dict[str, Any]:
    """Call LLM and parse action."""
    user_prompt = build_user_prompt(obs, history, last_reward)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        action = json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        # Fallback safe action
        action = {
            "urgency_level": "MEDIUM",
            "reasoning": "Unable to parse response, defaulting to medium urgency.",
            "department": "Emergency",
            "confidence": 0.3,
            "missing_info": [],
            "patient_ids": [],
        }
    return action


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(
    client: OpenAI,
    env: MedTriageHTTPEnv,
    task_id: str,
) -> Dict[str, Any]:
    """Run one full episode for a given task. Returns score summary."""
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_reward = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id)

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            action = get_model_action(client, obs, history, last_reward)
            action_summary = f"urgency={action.get('urgency_level')} dept={action.get('department')}"

            try:
                result = env.step(task_id, action)
                obs = result["observation"]
                reward = float(result.get("reward") or 0.0)
                done = result.get("done", False)
                error = None
            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_summary, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_summary!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception:
        pass
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required. Set it in your environment before running inference.py")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = MedTriageHTTPEnv(base_url=ENV_BASE_URL)

    results = []
    try:
        for task_id in TASKS:
            result = run_task(client, env, task_id)
            results.append(result)
    finally:
        env.close()


if __name__ == "__main__":
    main()