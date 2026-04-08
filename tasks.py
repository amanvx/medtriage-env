"""
MedTriage-Env: Task graders with semantic scoring.

Key improvements over v1:
- Semantic reasoning scoring (not just keyword matching)
- Fixed Task 2 patient_ids fallback bug
- Clinical trap detection (penalizes dangerous under-triage)
- Anti-gaming: reasoning length alone doesn't earn points
- Normalized ordering score handles partial patient_ids lists
"""

from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple

from models import TriageAction, TriageReward
from data import (
    TASK1_CASES, TASK2_QUEUES, TASK3_CASES,
    URGENCY_RANK, DEPARTMENT_KEYWORDS
)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _urgency_score(predicted: str, ground_truth: str) -> float:
    """
    Distance-based urgency scoring with partial credit.
    Exact = 1.0, 1 off = 0.65, 2 off = 0.35, 3+ = 0.0
    """
    p = predicted.upper().strip()
    g = ground_truth.upper().strip()
    if p not in URGENCY_RANK or g not in URGENCY_RANK:
        return 0.0
    if p == g:
        return 1.0
    diff = abs(URGENCY_RANK[p] - URGENCY_RANK[g])
    if diff == 1:
        return 0.65
    if diff == 2:
        return 0.35
    return 0.0


def _department_score(predicted: str, ground_truth: str) -> float:
    """Fuzzy department matching using keyword lookup."""
    pred_lower = predicted.lower()
    # Direct match
    if pred_lower == ground_truth.lower():
        return 1.0
    # Keyword match
    gt_keywords = DEPARTMENT_KEYWORDS.get(ground_truth, [ground_truth.lower()])
    for kw in gt_keywords:
        if kw in pred_lower:
            return 1.0
    # Partial: both point to urgent care even if different label
    emergency_words = ["emergency", "er", "ed", "acute", "resus", "icu"]
    if ground_truth == "Emergency" and any(w in pred_lower for w in emergency_words):
        return 1.0
    return 0.0


def _semantic_reasoning_score(reasoning: str, required_concepts: List[str]) -> float:
    """
    Score reasoning quality semantically:
    - Concept coverage: did the agent identify key clinical concepts?
    - Coherence: is there actual reasoning, not just word-stuffing?
    - Anti-gaming: penalize reasoning that is clearly just listing keywords
    """
    if not reasoning or len(reasoning.strip()) < 30:
        return 0.0

    text = reasoning.lower()

    # Detect keyword stuffing (many commas, very short phrases)
    word_count = len(text.split())
    comma_density = text.count(",") / max(word_count, 1)
    if comma_density > 0.15 and word_count < 40:
        # Likely just a list of keywords
        return 0.1

    # Concept coverage
    hits = 0
    for concept in required_concepts:
        concept_words = concept.lower().split()
        # All words of concept must appear in reasoning
        if all(w in text for w in concept_words):
            hits += 1
        # Partial: at least the main word
        elif concept_words[0] in text:
            hits += 0.4

    concept_score = min(1.0, hits / max(len(required_concepts), 1))

    # Coherence bonus: reasoning contains verbs (actual sentences)
    has_verbs = bool(re.search(
        r'\b(is|are|was|were|has|have|indicates|suggests|presents|requires|needs|shows|reveals)\b',
        text
    ))
    coherence_bonus = 0.15 if has_verbs else 0.0

    return round(min(1.0, 0.7 * concept_score + coherence_bonus + 0.15 * min(1.0, word_count / 60)), 3)


def _clinical_trap_penalty(predicted: str, trap_info: str, ground_truth: str) -> float:
    """
    Extra penalty for dangerous under-triage (e.g. marking CRITICAL as LOW).
    This catches the most clinically dangerous errors.
    """
    p = predicted.upper().strip()
    g = ground_truth.upper().strip()
    if g == "CRITICAL" and p in ["LOW", "NON_URGENT"]:
        return 0.3  # Serious clinical error
    if g == "CRITICAL" and p == "MEDIUM":
        return 0.15
    if g == "HIGH" and p == "NON_URGENT":
        return 0.15
    return 0.0


def _ordering_score(predicted_order: List[str], ground_truth_order: List[str]) -> float:
    """
    Kendall tau concordance: fraction of pairs in correct relative order.
    Handles partial lists (agent only ranked some patients).
    """
    # Only score patients that appear in both lists
    common = [p for p in ground_truth_order if p in predicted_order]
    if len(common) < 2:
        # Give partial credit if at least top-1 is right
        if predicted_order and predicted_order[0] == ground_truth_order[0]:
            return 0.3
        return 0.0

    pred_rank = {pid: i for i, pid in enumerate(predicted_order)}
    gt_rank = {pid: i for i, pid in enumerate(ground_truth_order)}

    concordant, total = 0, 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            total += 1
            if (gt_rank[a] < gt_rank[b]) == (pred_rank[a] < pred_rank[b]):
                concordant += 1

    return round(concordant / total, 3) if total > 0 else 0.0


def _missing_info_score(flagged: List[str], required: List[str], min_required: int) -> float:
    """
    Score missing info identification using token overlap.
    Partial credit for partially correct items.
    """
    if not flagged:
        return 0.0

    flagged_tokens = set()
    for f in flagged:
        flagged_tokens.update(f.lower().split())

    hits = 0.0
    for req in required:
        req_tokens = set(req.lower().split())
        # Remove stop words
        req_tokens -= {"the", "a", "an", "of", "and", "or", "in", "on"}
        if not req_tokens:
            continue
        overlap = len(req_tokens & flagged_tokens) / len(req_tokens)
        if overlap >= 0.6:
            hits += 1.0
        elif overlap >= 0.3:
            hits += 0.5

    coverage = min(1.0, hits / max(min_required, 1))
    # Bonus: flagging more than minimum required
    breadth_bonus = min(0.15, (hits - min_required) * 0.05) if hits > min_required else 0.0
    return round(min(1.0, coverage + breadth_bonus), 3)


# ---------------------------------------------------------------------------
# Task 1 Grader
# ---------------------------------------------------------------------------

def grade_task1(action: TriageAction, case_idx: int) -> TriageReward:
    case = TASK1_CASES[case_idx % len(TASK1_CASES)]
    gt = case["ground_truth"]

    urgency = _urgency_score(action.urgency_level, gt["urgency_level"])
    dept = _department_score(action.department, gt["department"])
    reasoning = _semantic_reasoning_score(action.reasoning, gt["key_reasoning"])

    # Clinical trap: dangerous under-triage
    trap_penalty = _clinical_trap_penalty(action.urgency_level, "", gt["urgency_level"])

    # Overconfidence penalty: wrong answer + high confidence
    overconfidence_penalty = 0.0
    if urgency < 0.35 and action.confidence > 0.8:
        overconfidence_penalty = 0.08

    penalty = min(0.4, trap_penalty + overconfidence_penalty)

    total = round(
        0.50 * urgency +
        0.20 * dept +
        0.30 * reasoning -
        penalty,
        3
    )
    total = max(0.0, min(1.0, total))

    return TriageReward(
        total=total,
        urgency_score=urgency,
        department_score=dept,
        reasoning_score=reasoning,
        penalty=penalty,
        explanation=(
            f"Urgency: {action.urgency_level} (GT:{gt['urgency_level']} score={urgency:.2f}) | "
            f"Dept: {action.department} (GT:{gt['department']} score={dept:.2f}) | "
            f"Reasoning={reasoning:.2f} | Penalty={penalty:.2f}"
        )
    )


# ---------------------------------------------------------------------------
# Task 2 Grader
# ---------------------------------------------------------------------------

def grade_task2(action: TriageAction, queue_idx: int = 0) -> TriageReward:
    queue = TASK2_QUEUES[queue_idx % len(TASK2_QUEUES)]
    gt_order = queue["ground_truth_order"]
    all_ids = [p.patient_id for p in queue["patients"]]

    # Validate patient_ids — only accept IDs that exist in this queue
    predicted_order = [pid for pid in action.patient_ids if pid in all_ids]

    # If agent returned no valid IDs, score 0 on ordering (no random fallback)
    ordering = _ordering_score(predicted_order, gt_order) if predicted_order else 0.0

    # Top-1 bonus: correctly identified most critical patient
    top1_bonus = 0.15 if (predicted_order and predicted_order[0] == gt_order[0]) else 0.0

    # Bottom penalty: placed most critical patient last
    bottom_penalty = 0.20 if (predicted_order and predicted_order[-1] == gt_order[0]) else 0.0

    # Reasoning quality
    top_patient = queue["patients"][[p.patient_id for p in queue["patients"]].index(gt_order[0])]
    key_concepts = top_patient.symptoms + [top_patient.chief_complaint.lower()]
    reasoning = _semantic_reasoning_score(action.reasoning, key_concepts)

    penalty = min(0.4, bottom_penalty)

    total = round(
        0.55 * ordering +
        top1_bonus +
        0.25 * reasoning -
        penalty,
        3
    )
    total = max(0.0, min(1.0, total))

    return TriageReward(
        total=total,
        ordering_score=ordering,
        reasoning_score=reasoning,
        penalty=penalty,
        explanation=(
            f"Ordering={ordering:.2f} | Top1 bonus={top1_bonus:.2f} | "
            f"Reasoning={reasoning:.2f} | Penalty={penalty:.2f} | "
            f"Predicted: {predicted_order} | GT: {gt_order}"
        )
    )


# ---------------------------------------------------------------------------
# Task 3 Grader
# ---------------------------------------------------------------------------

def grade_task3(action: TriageAction, case_idx: int) -> TriageReward:
    case = TASK3_CASES[case_idx % len(TASK3_CASES)]
    gt = case["ground_truth"]

    urgency = _urgency_score(action.urgency_level, gt["urgency_level"])
    dept = _department_score(action.department, gt["department"])

    missing = _missing_info_score(
        action.missing_info,
        gt["key_missing_info"],
        gt["min_missing_to_flag"]
    )

    # Clinical trap penalty — this is the hardest part
    trap_penalty = _clinical_trap_penalty(action.urgency_level, gt["trap"], gt["urgency_level"])

    # Overconfidence penalty — must flag uncertainty with incomplete data
    overconfidence_penalty = 0.0
    if action.confidence > 0.85 and len(action.missing_info) < 2:
        overconfidence_penalty = 0.20
    elif action.confidence > 0.75 and len(action.missing_info) < 2:
        overconfidence_penalty = 0.10

    # Reasoning must mention uncertainty/incompleteness
    uncertainty_concepts = ["missing", "incomplete", "unknown", "unavailable",
                             "cannot confirm", "needs", "required", "uncertain"] + gt["key_missing_info"][:3]
    reasoning = _semantic_reasoning_score(action.reasoning, uncertainty_concepts)

    penalty = min(0.5, trap_penalty + overconfidence_penalty)

    total = round(
        0.35 * urgency +
        0.15 * dept +
        0.35 * missing +
        0.15 * reasoning -
        penalty,
        3
    )
    total = max(0.0, min(1.0, total))

    return TriageReward(
        total=total,
        urgency_score=urgency,
        department_score=dept,
        missing_info_score=missing,
        reasoning_score=reasoning,
        penalty=penalty,
        explanation=(
            f"Urgency={urgency:.2f} | Dept={dept:.2f} | "
            f"Missing={missing:.2f} ({len(action.missing_info)} flagged) | "
            f"Reasoning={reasoning:.2f} | Penalty={penalty:.2f} | Trap: {gt['trap'][:60]}"
        )
    )


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "task1_single": {
        "name": "Single Patient Urgency Classification",
        "difficulty": "easy",
        "description": (
            "Classify a single patient's urgency level (CRITICAL/HIGH/MEDIUM/LOW/NON_URGENT), "
            "assign a target department, and provide clinical reasoning. "
            "Partial credit for correct urgency direction. Penalises dangerous under-triage."
        ),
        "max_steps": 5,
        "grader": grade_task1,
        "num_cases": len(TASK1_CASES),
    },
    "task2_queue": {
        "name": "Multi-Patient Queue Prioritization",
        "difficulty": "medium",
        "description": (
            "Given a waiting room of 5 patients, return their patient_ids ordered from "
            "highest to lowest priority. Justify your ranking. "
            "Scored using Kendall tau concordance + top-1 accuracy bonus."
        ),
        "max_steps": 5,
        "grader": grade_task2,
        "num_cases": len(TASK2_QUEUES),
    },
    "task3_incomplete": {
        "name": "Incomplete Notes — Flag, Reason, and Triage",
        "difficulty": "hard",
        "description": (
            "Patient notes have critical missing information. "
            "Identify ALL missing data items in missing_info. "
            "Assign urgency despite incomplete data. "
            "Use low confidence when data is absent. "
            "Penalises overconfidence and clinically dangerous undertriage traps."
        ),
        "max_steps": 7,
        "grader": grade_task3,
        "num_cases": len(TASK3_CASES),
    },
}