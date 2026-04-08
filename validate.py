"""
MedTriage-Env: Pre-submission validation script.
Run this before submitting to catch issues early.

Usage:
    python validate.py
"""

import sys
import json
import traceback

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"

errors = []
warnings = []


def check(name: str, fn):
    try:
        result = fn()
        if result is True or result is None:
            print(f"{PASS} {name}")
            return True
        elif result is False:
            print(f"{FAIL} {name}")
            errors.append(name)
            return False
        else:
            print(f"{PASS} {name}: {result}")
            return True
    except Exception as e:
        print(f"{FAIL} {name}: {e}")
        errors.append(f"{name}: {e}")
        return False


def warn(name: str, fn):
    try:
        fn()
        print(f"{PASS} {name}")
    except Exception as e:
        print(f"{WARN} {name}: {e}")
        warnings.append(name)


print("\n========================================")
print(" MedTriage-Env Pre-Submission Validator")
print("========================================\n")

# --- Imports ---
print("[1] Module imports")

def test_models():
    from models import TriageAction, TriageObservation, TriageReward, EnvironmentState, StepResult
    return True

def test_data():
    from data import TASK1_CASES, TASK2_QUEUES, TASK3_CASES
    assert len(TASK1_CASES) >= 10, f"Need >= 10 Task1 cases, got {len(TASK1_CASES)}"
    assert len(TASK2_QUEUES) >= 2, f"Need >= 2 queues, got {len(TASK2_QUEUES)}"
    assert len(TASK3_CASES) >= 3, f"Need >= 3 Task3 cases, got {len(TASK3_CASES)}"
    return f"Task1={len(TASK1_CASES)} Task2={len(TASK2_QUEUES)} queues Task3={len(TASK3_CASES)}"

def test_tasks():
    from tasks import TASK_REGISTRY
    assert "task1_single" in TASK_REGISTRY
    assert "task2_queue" in TASK_REGISTRY
    assert "task3_incomplete" in TASK_REGISTRY
    return f"{len(TASK_REGISTRY)} tasks registered"

def test_env():
    from environment import MedTriageEnv
    return True

check("models.py", test_models)
check("data.py", test_data)
check("tasks.py", test_tasks)
check("environment.py", test_env)

# --- OpenEnv API ---
print("\n[2] OpenEnv API compliance")

def test_reset_all_tasks():
    from environment import MedTriageEnv
    from models import TriageObservation
    for task_id in ["task1_single", "task2_queue", "task3_incomplete"]:
        env = MedTriageEnv(task_id=task_id)
        obs = env.reset()
        assert isinstance(obs, TriageObservation), f"reset() must return TriageObservation for {task_id}"
        assert obs.task_id == task_id
        assert len(obs.patients) > 0, f"Observation must have patients for {task_id}"
    return True

def test_step_returns_correct_types():
    from environment import MedTriageEnv
    from models import TriageAction, StepResult
    env = MedTriageEnv("task1_single")
    env.reset()
    action = TriageAction(
        urgency_level="CRITICAL",
        reasoning="Chest pain with ST elevation in diabetic hypertensive smoker — STEMI presentation.",
        department="Emergency",
        confidence=0.9
    )
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert 0.0 <= result.reward <= 1.0, f"Reward must be in [0,1], got {result.reward}"
    assert isinstance(result.done, bool)
    return f"reward={result.reward:.3f}"

def test_state():
    from environment import MedTriageEnv
    from models import EnvironmentState
    env = MedTriageEnv("task1_single")
    env.reset()
    s = env.state()
    assert isinstance(s, EnvironmentState)
    return True

def test_rewards_in_range():
    from environment import MedTriageEnv
    from models import TriageAction
    results = []
    for task_id in ["task1_single", "task2_queue", "task3_incomplete"]:
        env = MedTriageEnv(task_id)
        env.reset()
        for _ in range(3):
            if task_id == "task2_queue":
                action = TriageAction(
                    urgency_level="CRITICAL",
                    reasoning="Patient with SpO2 88% refractory to bronchodilators requires immediate intervention. Sudden vision loss is time-critical. Chronic pain is lowest priority.",
                    department="Emergency",
                    confidence=0.8,
                    patient_ids=["QA004", "QA002", "QA003", "QA005", "QA001"]
                )
            elif task_id == "task3_incomplete":
                action = TriageAction(
                    urgency_level="CRITICAL",
                    reasoning="Multiple critical data items are missing including blood pressure and oxygen saturation. Cannot confirm diagnosis without these values.",
                    department="Emergency",
                    confidence=0.4,
                    missing_info=["blood pressure", "oxygen saturation", "medical history", "medication list"]
                )
            else:
                action = TriageAction(
                    urgency_level="HIGH",
                    reasoning="Patient presents with significant symptoms requiring urgent evaluation.",
                    department="Emergency",
                    confidence=0.7
                )
            r = env.step(action)
            assert 0.0 <= r.reward <= 1.0, f"Reward {r.reward} out of range for {task_id}"
            results.append(r.reward)
    # Rewards must not all be identical (grader is not trivial)
    assert len(set(round(r, 2) for r in results)) > 1, "All rewards identical — grader may be broken"
    return f"Rewards: {[round(r,3) for r in results]}"

def test_done_flag():
    from environment import MedTriageEnv
    from models import TriageAction
    env = MedTriageEnv("task1_single")
    env.reset()
    action = TriageAction(urgency_level="HIGH", reasoning="Test action for validation purposes.", department="Emergency", confidence=0.5)
    for i in range(env._max_steps):
        result = env.step(action)
    assert result.done is True, "done must be True after max_steps"
    return f"done=True after {env._max_steps} steps"

check("reset() all tasks", test_reset_all_tasks)
check("step() return types", test_step_returns_correct_types)
check("state() return type", test_state)
check("rewards in [0,1] range", test_rewards_in_range)
check("done flag after max_steps", test_done_flag)

# --- Grader quality ---
print("\n[3] Grader quality checks")

def test_graders_not_constant():
    """Graders must return different scores for different actions."""
    from tasks import grade_task1
    from models import TriageAction

    perfect = TriageAction(
        urgency_level="CRITICAL",
        reasoning="ST elevation myocardial infarction with diaphoresis and chest pain in diabetic hypertensive smoker. Immediate cath lab activation required.",
        department="Emergency",
        confidence=0.95
    )
    wrong = TriageAction(
        urgency_level="NON_URGENT",
        reasoning="Mild discomfort, can wait.",
        department="General",
        confidence=0.9
    )
    r_perfect = grade_task1(perfect, 0).total
    r_wrong = grade_task1(wrong, 0).total
    assert r_perfect != r_wrong, "Grader returns same score for good and bad answers"
    assert r_perfect > r_wrong, f"Perfect answer ({r_perfect}) should score higher than wrong ({r_wrong})"
    return f"Perfect={r_perfect:.3f} > Wrong={r_wrong:.3f}"

def test_task2_grader_no_random():
    """Task 2 must not fallback to random ordering."""
    from tasks import grade_task2
    from models import TriageAction
    action_no_ids = TriageAction(
        urgency_level="CRITICAL",
        reasoning="Ordering patients by severity.",
        department="Emergency",
        confidence=0.5,
        patient_ids=[]  # Empty — no random fallback allowed
    )
    r = grade_task2(action_no_ids, 0)
    # Should get 0 ordering score, not a random score
    assert r.ordering_score == 0.0, f"Empty patient_ids should give 0 ordering, got {r.ordering_score}"
    return f"Empty ids -> ordering=0.0 (correct)"

def test_task3_overconfidence_penalty():
    """High confidence + no missing_info = penalty."""
    from tasks import grade_task3
    from models import TriageAction
    overconfident = TriageAction(
        urgency_level="CRITICAL",
        reasoning="I am certain this patient needs emergency care immediately.",
        department="Emergency",
        confidence=0.95,
        missing_info=[]  # Should be penalized
    )
    humble = TriageAction(
        urgency_level="CRITICAL",
        reasoning="Blood pressure and oxygen saturation are unavailable. Medical history and medication list are unknown. Cannot confirm without these data items.",
        department="Emergency",
        confidence=0.4,
        missing_info=["blood pressure", "oxygen saturation", "medical history", "medication list"]
    )
    r_overconfident = grade_task3(overconfident, 0).total
    r_humble = grade_task3(humble, 0).total
    assert r_humble > r_overconfident, f"Humble ({r_humble}) should outscore overconfident ({r_overconfident})"
    return f"Humble={r_humble:.3f} > Overconfident={r_overconfident:.3f}"

check("graders not constant", test_graders_not_constant)
check("task2 no random fallback", test_task2_grader_no_random)
check("task3 overconfidence penalty", test_task3_overconfidence_penalty)

# --- Files ---
print("\n[4] Required files")

import os
REQUIRED_FILES = [
    "models.py", "data.py", "tasks.py", "environment.py",
    "app.py", "inference.py", "openenv.yaml", "Dockerfile",
    "requirements.txt", "README.md"
]
for f in REQUIRED_FILES:
    check(f, lambda f=f: os.path.exists(f))

# --- YAML ---
print("\n[5] openenv.yaml")

def test_yaml():
    import yaml
    with open("openenv.yaml") as fh:
        cfg = yaml.safe_load(fh)
    assert "name" in cfg
    assert "tasks" in cfg
    assert len(cfg["tasks"]) >= 3, f"Need >= 3 tasks in yaml, got {len(cfg['tasks'])}"
    return f"name={cfg['name']} tasks={len(cfg['tasks'])}"

warn("openenv.yaml valid", test_yaml)

# --- Summary ---
print("\n========================================")
if errors:
    print(f" FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"   - {e}")
    print(" Fix these before submitting!")
    sys.exit(1)
else:
    print(f" All checks passed!")
    if warnings:
        print(f" {len(warnings)} warning(s) — review before submitting")
    print(" Your submission looks ready.")
print("========================================\n")
