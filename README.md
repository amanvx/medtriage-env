---
title: MedTriage-Env
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - medical
  - triage
  - healthcare
  - reinforcement-learning
  - agent-environment
license: mit
---

# 🏥 MedTriage-Env

**A clinical note triage OpenEnv environment for AI agents.**

MedTriage-Env simulates real emergency department and hospital triage workflows. Given patient clinical notes, an AI agent must assign urgency levels, route patients to the correct departments, and identify critical missing information — exactly as a triage nurse or physician would.

---

## 🌍 Real-World Motivation

Hospital triage is performed millions of times daily worldwide. Errors cost lives: under-triaging critical patients causes preventable deaths, while over-triaging wastes scarce resources. AI agents that can learn to triage from clinical notes have immediate, direct applications in:

- Emergency department decision support
- Telehealth pre-screening systems
- Hospital queue management
- Medical AI safety evaluation

MedTriage-Env fills a genuine gap in the OpenEnv ecosystem — no medical triage environment exists yet.

---

## 🗂️ Tasks

### Task 1: Single Patient Urgency Classification *(Easy)*
**Objective:** Given one patient's clinical note, classify urgency and route to department.
- **Baseline score:** ~0.72

### Task 2: Multi-Patient Queue Prioritization *(Medium)*
**Objective:** Given 5 waiting patients, order them from highest to lowest priority.
- **Baseline score:** ~0.58

### Task 3: Incomplete Notes — Flag and Triage *(Hard)*
**Objective:** Triage patients with deliberately missing clinical data. Flag what's missing AND assign urgency despite uncertainty.
- **Baseline score:** ~0.44

---

## 📐 Observation Space

```json
{
  "task_id": "task1_single | task2_queue | task3_incomplete",
  "task_description": "Natural language task instructions",
  "patients": [{"patient_id": "P001", "chief_complaint": "...", "vitals": {}, "history": "...", "symptoms": [], "duration": "...", "age": 58, "sex": "M"}],
  "step_number": 0,
  "max_steps": 5,
  "feedback": "",
  "cumulative_reward": 0.0,
  "done": false
}
```

## ⚡ Action Space

```json
{
  "urgency_level": "CRITICAL | HIGH | MEDIUM | LOW | NON_URGENT",
  "reasoning": "Clinical reasoning (2-4 sentences)",
  "department": "Emergency | Cardiology | Pulmonology | Orthopedics | Neurology | General",
  "confidence": 0.85,
  "missing_info": ["blood pressure", "oxygen saturation"],
  "patient_ids": ["Q004", "Q002", "Q001", "Q005", "Q003"]
}
```

## 🎯 Reward Function

Shaped reward with partial credit throughout the trajectory (not just at episode end).

| Component | Task 1 | Task 2 | Task 3 |
|-----------|--------|--------|--------|
| Urgency accuracy | 50% | — | 35% |
| Department routing | 20% | — | 15% |
| Reasoning quality | 30% | 25% | 15% |
| Queue ordering (Kendall tau) | — | 55% | — |
| Top-1 priority bonus | — | +15% | — |
| Missing info coverage | — | — | 35% |
| Overconfidence penalty | up to −10% | up to −20% | up to −20% |

---

## 🚀 Setup & Usage

### Docker
```bash
docker build -t medtriage-env .
docker run -p 7860:7860 medtriage-env
```

### Local
```bash
pip install -r requirements.txt
python app.py
```

### Run Baseline Inference
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_key_here
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

---

## 📊 Baseline Scores

| Task | Score | Difficulty |
|------|-------|------------|
| task1_single | 0.72 | Easy |
| task2_queue | 0.58 | Medium |
| task3_incomplete | 0.44 | Hard |
| **Average** | **0.58** | — |

---

## ⚠️ Disclaimer

All patient data is entirely synthetic and randomly generated for AI training purposes only.

## 📄 License

MIT License
