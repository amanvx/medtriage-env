"""
MedTriage-Env: Synthetic patient clinical cases.
30+ cases across all tasks. All data is fictional — AI training only.
"""

from __future__ import annotations
from typing import Dict, Any, List
from models import PatientNote

# ---------------------------------------------------------------------------
# TASK 1 — Single patient cases (15 cases, varied difficulty)
# ---------------------------------------------------------------------------

TASK1_CASES: List[Dict[str, Any]] = [
    {
        "patient": PatientNote(
            patient_id="P001",
            chief_complaint="Crushing chest pain radiating to left arm",
            vitals={"bp": "180/110", "hr": 112, "rr": 22, "spo2": 94, "temp": 37.1},
            history="Hypertension, type 2 diabetes, smoker 20 years",
            symptoms=["chest pain", "diaphoresis", "nausea", "left arm pain"],
            duration="45 minutes", age=58, sex="M",
            notes="Pale and diaphoretic. ECG shows ST elevation V2-V4."
        ),
        "ground_truth": {"urgency_level": "CRITICAL", "department": "Emergency",
                         "key_reasoning": ["STEMI", "ST elevation", "chest pain", "diaphoresis", "cardiac"]}
    },
    {
        "patient": PatientNote(
            patient_id="P002",
            chief_complaint="Mild sore throat and runny nose",
            vitals={"bp": "118/76", "hr": 72, "rr": 14, "spo2": 99, "temp": 37.4},
            history="No significant medical history",
            symptoms=["sore throat", "rhinorrhea", "mild fatigue"],
            duration="2 days", age=24, sex="F",
            notes="No difficulty swallowing, no fever, no stridor."
        ),
        "ground_truth": {"urgency_level": "NON_URGENT", "department": "General",
                         "key_reasoning": ["viral", "upper respiratory", "stable", "no fever"]}
    },
    {
        "patient": PatientNote(
            patient_id="P003",
            chief_complaint="Sudden severe headache, worst of life",
            vitals={"bp": "162/98", "hr": 88, "rr": 16, "spo2": 97, "temp": 37.2},
            history="No prior headaches of this severity",
            symptoms=["thunderclap headache", "neck stiffness", "photophobia", "vomiting"],
            duration="1 hour", age=41, sex="F",
            notes="10/10 pain, sudden onset. Meningismus present on exam."
        ),
        "ground_truth": {"urgency_level": "CRITICAL", "department": "Emergency",
                         "key_reasoning": ["subarachnoid", "thunderclap", "meningismus", "worst headache"]}
    },
    {
        "patient": PatientNote(
            patient_id="P004",
            chief_complaint="Ankle pain after twisting while playing football",
            vitals={"bp": "122/78", "hr": 80, "rr": 15, "spo2": 99, "temp": 36.8},
            history="No significant history",
            symptoms=["ankle pain", "swelling", "bruising"],
            duration="3 hours", age=19, sex="M",
            notes="Can bear weight with difficulty. Tenderness over lateral malleolus. No neurovascular deficit."
        ),
        "ground_truth": {"urgency_level": "LOW", "department": "Orthopedics",
                         "key_reasoning": ["sprain", "weight bearing", "lateral malleolus", "stable"]}
    },
    {
        "patient": PatientNote(
            patient_id="P005",
            chief_complaint="Fever and productive cough",
            vitals={"bp": "128/82", "hr": 98, "rr": 20, "spo2": 95, "temp": 38.9},
            history="COPD, on inhalers",
            symptoms=["fever", "productive cough", "dyspnea", "fatigue"],
            duration="4 days", age=67, sex="M",
            notes="Yellow-green sputum. Increased inhaler use. Crackles right lower lobe."
        ),
        "ground_truth": {"urgency_level": "HIGH", "department": "Pulmonology",
                         "key_reasoning": ["COPD exacerbation", "pneumonia", "crackles", "SpO2 95"]}
    },
    {
        "patient": PatientNote(
            patient_id="P006",
            chief_complaint="Severe allergic reaction after bee sting",
            vitals={"bp": "82/50", "hr": 128, "rr": 26, "spo2": 91, "temp": 37.0},
            history="Known bee allergy, EpiPen not available",
            symptoms=["urticaria", "throat swelling", "wheezing", "hypotension", "dizziness"],
            duration="15 minutes", age=33, sex="F",
            notes="Stridor present. Angioedema of lips and tongue. Deteriorating rapidly."
        ),
        "ground_truth": {"urgency_level": "CRITICAL", "department": "Emergency",
                         "key_reasoning": ["anaphylaxis", "stridor", "angioedema", "hypotension", "epinephrine"]}
    },
    {
        "patient": PatientNote(
            patient_id="P007",
            chief_complaint="Blood sugar 38 mg/dL, found confused at home",
            vitals={"bp": "138/86", "hr": 102, "rr": 18, "spo2": 98, "temp": 36.9},
            history="Type 1 diabetes on insulin pump",
            symptoms=["confusion", "diaphoresis", "tremor", "weakness"],
            duration="30 minutes", age=29, sex="M",
            notes="BGL 38 confirmed. Responsive but confused. Family gave juice — minimal improvement."
        ),
        "ground_truth": {"urgency_level": "CRITICAL", "department": "Emergency",
                         "key_reasoning": ["hypoglycemia", "BGL 38", "confusion", "IV dextrose needed"]}
    },
    {
        "patient": PatientNote(
            patient_id="P008",
            chief_complaint="Chronic knee pain, requesting physiotherapy referral",
            vitals={"bp": "126/80", "hr": 70, "rr": 14, "spo2": 99, "temp": 36.6},
            history="Osteoarthritis, well-controlled",
            symptoms=["knee pain", "stiffness", "reduced range of motion"],
            duration="6 months, stable", age=72, sex="F",
            notes="No acute swelling or warmth. No trauma. Stable on NSAIDs."
        ),
        "ground_truth": {"urgency_level": "NON_URGENT", "department": "Orthopedics",
                         "key_reasoning": ["chronic", "stable", "osteoarthritis", "no acute change"]}
    },
    {
        "patient": PatientNote(
            patient_id="P009",
            chief_complaint="Facial drooping and arm weakness, sudden onset",
            vitals={"bp": "192/108", "hr": 84, "rr": 16, "spo2": 97, "temp": 37.0},
            history="Atrial fibrillation, on warfarin",
            symptoms=["facial droop", "arm weakness", "slurred speech", "confusion"],
            duration="40 minutes", age=71, sex="M",
            notes="FAST positive. Last known well 40 minutes ago. On anticoagulation."
        ),
        "ground_truth": {"urgency_level": "CRITICAL", "department": "Neurology",
                         "key_reasoning": ["stroke", "FAST", "thrombolysis window", "atrial fibrillation"]}
    },
    {
        "patient": PatientNote(
            patient_id="P010",
            chief_complaint="Mild urinary burning and frequency",
            vitals={"bp": "118/74", "hr": 74, "rr": 14, "spo2": 99, "temp": 37.3},
            history="Recurrent UTIs, otherwise healthy",
            symptoms=["dysuria", "urinary frequency", "mild suprapubic discomfort"],
            duration="2 days", age=28, sex="F",
            notes="No fever. No flank pain. Urine dipstick positive nitrites."
        ),
        "ground_truth": {"urgency_level": "LOW", "department": "General",
                         "key_reasoning": ["uncomplicated UTI", "no fever", "no pyelonephritis", "stable"]}
    },
    {
        "patient": PatientNote(
            patient_id="P011",
            chief_complaint="Severe abdominal pain, rigid abdomen",
            vitals={"bp": "98/62", "hr": 118, "rr": 24, "spo2": 96, "temp": 38.8},
            history="Peptic ulcer disease",
            symptoms=["severe abdominal pain", "rigid abdomen", "guarding", "rebound tenderness", "fever"],
            duration="2 hours", age=52, sex="M",
            notes="Board-like rigidity. Peritoneal signs positive. Suspected perforation."
        ),
        "ground_truth": {"urgency_level": "CRITICAL", "department": "Emergency",
                         "key_reasoning": ["peritonitis", "perforation", "rigid abdomen", "surgical emergency"]}
    },
    {
        "patient": PatientNote(
            patient_id="P012",
            chief_complaint="Palpitations and racing heart",
            vitals={"bp": "136/84", "hr": 168, "rr": 18, "spo2": 97, "temp": 37.0},
            history="No cardiac history",
            symptoms=["palpitations", "mild dizziness", "chest fluttering"],
            duration="20 minutes", age=35, sex="F",
            notes="Regular rhythm on monitor. No chest pain. No syncope. Hemodynamically stable."
        ),
        "ground_truth": {"urgency_level": "HIGH", "department": "Cardiology",
                         "key_reasoning": ["tachyarrhythmia", "HR 168", "hemodynamically stable", "needs ECG"]}
    },
    {
        "patient": PatientNote(
            patient_id="P013",
            chief_complaint="Routine blood pressure check",
            vitals={"bp": "134/86", "hr": 68, "rr": 14, "spo2": 99, "temp": 36.7},
            history="Hypertension, well controlled on medication",
            symptoms=["no symptoms"],
            duration="Routine visit", age=55, sex="M",
            notes="Asymptomatic. BP slightly above target. No hypertensive crisis."
        ),
        "ground_truth": {"urgency_level": "NON_URGENT", "department": "General",
                         "key_reasoning": ["routine", "asymptomatic", "controlled hypertension"]}
    },
    {
        "patient": PatientNote(
            patient_id="P014",
            chief_complaint="Deep laceration to forearm from glass",
            vitals={"bp": "124/78", "hr": 86, "rr": 16, "spo2": 98, "temp": 36.8},
            history="No significant history. Tetanus up to date.",
            symptoms=["laceration", "bleeding", "pain"],
            duration="1 hour", age=22, sex="M",
            notes="5cm deep laceration, actively bleeding. Pressure applied. Neurovascular intact distally."
        ),
        "ground_truth": {"urgency_level": "MEDIUM", "department": "Emergency",
                         "key_reasoning": ["laceration", "active bleeding", "sutures needed"]}
    },
    {
        "patient": PatientNote(
            patient_id="P015",
            chief_complaint="Difficulty breathing, unable to speak in full sentences",
            vitals={"bp": "142/90", "hr": 122, "rr": 32, "spo2": 86, "temp": 37.4},
            history="Severe asthma, multiple ICU admissions",
            symptoms=["severe dyspnea", "inability to complete sentences", "accessory muscle use", "silent chest"],
            duration="1 hour, worsening", age=26, sex="F",
            notes="Silent chest on auscultation. Nebuliser x4 — no improvement. Near-fatal asthma history."
        ),
        "ground_truth": {"urgency_level": "CRITICAL", "department": "Emergency",
                         "key_reasoning": ["status asthmaticus", "silent chest", "SpO2 86", "near-fatal"]}
    },
]

# ---------------------------------------------------------------------------
# TASK 2 — Multiple queues (4 queues for variety)
# ---------------------------------------------------------------------------

TASK2_QUEUES: List[Dict[str, Any]] = [
    {
        "queue_id": "QA",
        "patients": [
            PatientNote(patient_id="QA001", chief_complaint="Lower back pain, chronic",
                vitals={"bp": "124/80", "hr": 74, "rr": 14, "spo2": 99, "temp": 36.7},
                history="Chronic back pain, physiotherapy ongoing",
                symptoms=["lower back pain", "stiffness"], duration="3 weeks", age=45, sex="M",
                notes="Rates pain 4/10, same as usual"),
            PatientNote(patient_id="QA002", chief_complaint="Sudden vision loss in right eye",
                vitals={"bp": "148/92", "hr": 86, "rr": 16, "spo2": 98, "temp": 37.0},
                history="Type 2 diabetes, 15 years",
                symptoms=["sudden vision loss", "no pain"], duration="2 hours", age=55, sex="F",
                notes="Curtain coming down over vision. Time-critical retinal event."),
            PatientNote(patient_id="QA003", chief_complaint="Allergic reaction, hives on arms",
                vitals={"bp": "116/74", "hr": 88, "rr": 16, "spo2": 99, "temp": 37.1},
                history="Known shellfish allergy",
                symptoms=["urticaria", "mild itching"], duration="1 hour", age=28, sex="F",
                notes="No throat swelling, no breathing difficulty"),
            PatientNote(patient_id="QA004", chief_complaint="Difficulty breathing, wheezing",
                vitals={"bp": "138/88", "hr": 118, "rr": 28, "spo2": 88, "temp": 37.3},
                history="Asthma, on salbutamol",
                symptoms=["wheezing", "dyspnea", "chest tightness", "accessory muscle use"],
                duration="30 minutes", age=32, sex="M",
                notes="Inhaler x3 — no relief. SpO2 dropping."),
            PatientNote(patient_id="QA005", chief_complaint="Urinary burning and frequency",
                vitals={"bp": "120/76", "hr": 76, "rr": 14, "spo2": 99, "temp": 37.6},
                history="Recurrent UTIs",
                symptoms=["dysuria", "urinary frequency"], duration="2 days", age=31, sex="F",
                notes="No fever, no flank pain"),
        ],
        "ground_truth_order": ["QA004", "QA002", "QA003", "QA005", "QA001"],
        "rationale": {"QA004": "CRITICAL - SpO2 88, refractory bronchospasm",
                      "QA002": "HIGH - sudden vision loss in diabetic, retinal artery occlusion window",
                      "QA003": "MEDIUM - allergic but no anaphylaxis",
                      "QA005": "LOW - uncomplicated UTI",
                      "QA001": "LOW - chronic stable pain"}
    },
    {
        "queue_id": "QB",
        "patients": [
            PatientNote(patient_id="QB001", chief_complaint="Mild headache and fatigue",
                vitals={"bp": "128/82", "hr": 72, "rr": 14, "spo2": 99, "temp": 37.0},
                history="Tension headaches, office worker",
                symptoms=["headache", "fatigue", "mild neck tension"], duration="1 day", age=38, sex="F",
                notes="Similar to previous tension headaches. No red flags."),
            PatientNote(patient_id="QB002", chief_complaint="Chest pain and irregular heartbeat",
                vitals={"bp": "156/96", "hr": 44, "rr": 18, "spo2": 96, "temp": 37.1},
                history="Known complete heart block, pacemaker dependent",
                symptoms=["chest pain", "dizziness", "near-syncope"],
                duration="45 minutes", age=68, sex="M",
                notes="Pacemaker may be malfunctioning. Bradycardia at 44bpm."),
            PatientNote(patient_id="QB003", chief_complaint="Sprained wrist from fall",
                vitals={"bp": "118/76", "hr": 78, "rr": 15, "spo2": 99, "temp": 36.8},
                history="No significant history",
                symptoms=["wrist pain", "swelling"], duration="2 hours", age=16, sex="M",
                notes="Mild swelling. Tenderness anatomical snuffbox. Neurovascular intact."),
            PatientNote(patient_id="QB004", chief_complaint="Sudden confusion and one-sided weakness",
                vitals={"bp": "198/114", "hr": 92, "rr": 18, "spo2": 96, "temp": 37.2},
                history="Hypertension, non-compliant with medications",
                symptoms=["confusion", "right arm weakness", "facial droop", "dysarthria"],
                duration="25 minutes", age=64, sex="M",
                notes="FAST positive. Within thrombolysis window."),
            PatientNote(patient_id="QB005", chief_complaint="Medication refill request",
                vitals={"bp": "138/88", "hr": 70, "rr": 14, "spo2": 99, "temp": 36.7},
                history="Hypertension, stable",
                symptoms=["no symptoms"], duration="Administrative", age=58, sex="F",
                notes="Stable, asymptomatic. Administrative visit."),
        ],
        "ground_truth_order": ["QB004", "QB002", "QB001", "QB003", "QB005"],
        "rationale": {"QB004": "CRITICAL - acute stroke, thrombolysis window closing",
                      "QB002": "CRITICAL - pacemaker failure, HR 44",
                      "QB001": "LOW - tension headache, stable",
                      "QB003": "LOW - minor wrist injury",
                      "QB005": "NON_URGENT - administrative refill"}
    },
    {
        "queue_id": "QC",
        "patients": [
            PatientNote(patient_id="QC001", chief_complaint="Severe right lower quadrant pain",
                vitals={"bp": "132/84", "hr": 104, "rr": 20, "spo2": 97, "temp": 38.4},
                history="No previous abdominal surgery",
                symptoms=["RLQ pain", "nausea", "vomiting", "fever", "rebound tenderness"],
                duration="8 hours, worsening", age=22, sex="M",
                notes="Rovsing sign positive. McBurney point tenderness. Suspected appendicitis."),
            PatientNote(patient_id="QC002", chief_complaint="Paper cut on finger",
                vitals={"bp": "116/74", "hr": 68, "rr": 14, "spo2": 99, "temp": 36.6},
                history="No significant history",
                symptoms=["minor finger laceration"], duration="30 minutes", age=30, sex="F",
                notes="Superficial. Bleeding controlled. No tendon involvement."),
            PatientNote(patient_id="QC003", chief_complaint="Vomiting blood",
                vitals={"bp": "94/58", "hr": 126, "rr": 22, "spo2": 95, "temp": 37.1},
                history="Alcohol use disorder, cirrhosis",
                symptoms=["hematemesis", "dizziness", "weakness"],
                duration="1 hour", age=48, sex="M",
                notes="Large volume bright red blood. Suspected variceal bleed. Hypotensive."),
            PatientNote(patient_id="QC004", chief_complaint="Diabetic foot wound, looks infected",
                vitals={"bp": "136/86", "hr": 88, "rr": 16, "spo2": 97, "temp": 38.1},
                history="Type 2 diabetes, peripheral vascular disease",
                symptoms=["foot wound", "swelling", "redness", "foul smell"],
                duration="5 days", age=61, sex="M",
                notes="Wagner grade 3 wound. Spreading cellulitis. No crepitus."),
            PatientNote(patient_id="QC005", chief_complaint="Insomnia and mild anxiety",
                vitals={"bp": "122/78", "hr": 76, "rr": 14, "spo2": 99, "temp": 36.8},
                history="Generalised anxiety disorder",
                symptoms=["insomnia", "anxiety", "mild palpitations"],
                duration="2 weeks", age=34, sex="F",
                notes="Stable vitals. No acute distress. Medication review needed."),
        ],
        "ground_truth_order": ["QC003", "QC001", "QC004", "QC005", "QC002"],
        "rationale": {"QC003": "CRITICAL - active upper GI bleed, hypotensive, variceal",
                      "QC001": "HIGH - suspected appendicitis, surgical evaluation urgent",
                      "QC004": "MEDIUM - infected diabetic foot",
                      "QC005": "LOW - chronic anxiety, stable",
                      "QC002": "NON_URGENT - paper cut, superficial"}
    },
    {
        "queue_id": "QD",
        "patients": [
            PatientNote(patient_id="QD001", chief_complaint="Toddler with high fever and rash",
                vitals={"bp": "N/A", "hr": 168, "rr": 42, "spo2": 94, "temp": 40.1},
                history="18-month-old, vaccinations up to date",
                symptoms=["fever 40.1", "non-blanching petechial rash", "irritability", "neck stiffness"],
                duration="4 hours", age=1, sex="M",
                notes="Non-blanching rash spreading. Bulging fontanelle. Meningococcal sepsis suspected."),
            PatientNote(patient_id="QD002", chief_complaint="32-week pregnant, reduced fetal movement",
                vitals={"bp": "168/108", "hr": 96, "rr": 18, "spo2": 98, "temp": 37.0},
                history="First pregnancy, gestational hypertension",
                symptoms=["reduced fetal movement", "headache", "visual disturbance", "epigastric pain"],
                duration="6 hours", age=28, sex="F",
                notes="BP 168/108. Proteinuria 3+. Severe preeclampsia. Fetal CTG abnormal."),
            PatientNote(patient_id="QD003", chief_complaint="Child fell off swing, arm pain",
                vitals={"bp": "N/A", "hr": 110, "rr": 22, "spo2": 99, "temp": 36.9},
                history="5-year-old, no significant history",
                symptoms=["left arm pain", "swelling", "refusing to move arm"],
                duration="1 hour", age=5, sex="F",
                notes="Deformity of forearm. Neurovascular intact distally. Likely fracture."),
            PatientNote(patient_id="QD004", chief_complaint="Nappy rash, mild",
                vitals={"bp": "N/A", "hr": 120, "rr": 30, "spo2": 99, "temp": 37.1},
                history="8-month-old, otherwise well",
                symptoms=["nappy rash", "mild crying"],
                duration="3 days", age=0, sex="M",
                notes="Mild erythema only. Feeding well. No systemic signs."),
            PatientNote(patient_id="QD005", chief_complaint="Teen with worsening asthma symptoms",
                vitals={"bp": "118/76", "hr": 102, "rr": 24, "spo2": 93, "temp": 37.2},
                history="Moderate persistent asthma",
                symptoms=["wheeze", "cough", "dyspnea", "reduced exercise tolerance"],
                duration="2 days, worsening", age=14, sex="F",
                notes="SpO2 93%. Accessory muscle use. Partially responsive to salbutamol."),
        ],
        "ground_truth_order": ["QD001", "QD002", "QD005", "QD003", "QD004"],
        "rationale": {"QD001": "CRITICAL - meningococcal sepsis, non-blanching rash",
                      "QD002": "CRITICAL - severe preeclampsia, abnormal CTG",
                      "QD005": "HIGH - SpO2 93%, worsening asthma in child",
                      "QD003": "MEDIUM - forearm fracture, stable neurovascular",
                      "QD004": "NON_URGENT - mild nappy rash"}
    },
]

# For backwards compat
TASK2_QUEUE = TASK2_QUEUES[0]

# ---------------------------------------------------------------------------
# TASK 3 — Incomplete cases with clinical traps (6 cases)
# ---------------------------------------------------------------------------

TASK3_CASES: List[Dict[str, Any]] = [
    {
        "patient": PatientNote(
            patient_id="I001",
            chief_complaint="Chest discomfort and fatigue",
            vitals={"bp": "UNAVAILABLE", "hr": 104, "rr": 18, "spo2": "UNAVAILABLE", "temp": 37.0},
            history="Unknown — patient unresponsive to history questions",
            symptoms=["chest discomfort", "fatigue", "dizziness"],
            duration="Unknown", age=62, sex="M",
            notes="Arrived alone, confused. Diaphoretic. No medications listed."
        ),
        "ground_truth": {
            "urgency_level": "CRITICAL", "department": "Emergency",
            "key_missing_info": ["blood pressure", "oxygen saturation", "medical history",
                                  "medication list", "symptom duration", "pain character",
                                  "ECG", "blood glucose"],
            "min_missing_to_flag": 4,
            "trap": "Diaphoresis + chest discomfort + confusion = CRITICAL even without full vitals"
        }
    },
    {
        "patient": PatientNote(
            patient_id="I002",
            chief_complaint="Abdominal pain",
            vitals={"bp": "110/70", "hr": 96, "rr": 18, "spo2": 97, "temp": 38.1},
            history="Previous abdominal surgeries — details unknown",
            symptoms=["abdominal pain", "nausea"],
            duration="Several hours", age=34, sex="F",
            notes="LMP unknown. Pregnancy status unknown. Pain location not specified."
        ),
        "ground_truth": {
            "urgency_level": "HIGH", "department": "Emergency",
            "key_missing_info": ["pregnancy status", "LMP", "pain location and character",
                                  "surgical history details", "vaginal bleeding",
                                  "shoulder tip pain", "bowel movements"],
            "min_missing_to_flag": 4,
            "trap": "Reproductive-age female + unknown pregnancy + abdominal pain = must rule out ectopic"
        }
    },
    {
        "patient": PatientNote(
            patient_id="I003",
            chief_complaint="Altered mental status",
            vitals={"bp": "90/60", "hr": 122, "rr": 24, "spo2": 93, "temp": 39.4},
            history="Unknown",
            symptoms=["confusion", "agitation"],
            duration="Found this way by family", age=78, sex="M",
            notes="No medication list. No allergy documentation. Baseline mental status unknown."
        ),
        "ground_truth": {
            "urgency_level": "CRITICAL", "department": "Emergency",
            "key_missing_info": ["medication list", "allergy history", "baseline mental status",
                                  "recent illness or trauma", "blood glucose",
                                  "urine output", "last oral intake"],
            "min_missing_to_flag": 4,
            "trap": "Septic shock picture (HR122, RR24, Temp39.4, BP90/60, AMS) = CRITICAL not HIGH"
        }
    },
    {
        "patient": PatientNote(
            patient_id="I004",
            chief_complaint="Back pain",
            vitals={"bp": "88/54", "hr": 118, "rr": 22, "spo2": 96, "temp": 36.9},
            history="Known AAA under surveillance, last documented size 2 years ago",
            symptoms=["severe back pain", "abdominal pain"],
            duration="Sudden onset 2 hours ago", age=74, sex="M",
            notes="AAA current size unknown. No recent imaging. Pulsatile mass palpated. Tender flanks."
        ),
        "ground_truth": {
            "urgency_level": "CRITICAL", "department": "Emergency",
            "key_missing_info": ["current AAA size", "recent imaging", "coagulation status",
                                  "blood type and crossmatch", "urine output", "distal pulses"],
            "min_missing_to_flag": 3,
            "trap": "Hypotension + pulsatile mass + sudden back pain = ruptured AAA until proven otherwise"
        }
    },
    {
        "patient": PatientNote(
            patient_id="I005",
            chief_complaint="Feeling unwell",
            vitals={"bp": "146/92", "hr": 88, "rr": 16, "spo2": 97, "temp": 37.3},
            history="Psychiatric history, on multiple medications — list unavailable",
            symptoms=["feeling unwell", "mild confusion", "nausea"],
            duration="Several hours", age=45, sex="F",
            notes="Brought in by mental health worker. Possible overdose not confirmed."
        ),
        "ground_truth": {
            "urgency_level": "HIGH", "department": "Emergency",
            "key_missing_info": ["medication list", "overdose history", "specific drug ingested",
                                  "last dose taken", "ECG", "paracetamol level",
                                  "salicylate level", "GCS"],
            "min_missing_to_flag": 4,
            "trap": "Possible intentional overdose — cannot be LOW/MEDIUM; toxicology workup critical"
        }
    },
    {
        "patient": PatientNote(
            patient_id="I006",
            chief_complaint="Shortness of breath post-surgery",
            vitals={"bp": "134/86", "hr": 110, "rr": 26, "spo2": 91, "temp": 37.8},
            history="Hip replacement 3 days ago",
            symptoms=["dyspnea", "pleuritic chest pain", "tachycardia"],
            duration="4 hours", age=66, sex="F",
            notes="Post-op day 3. DVT prophylaxis status unknown. No Wells score documented."
        ),
        "ground_truth": {
            "urgency_level": "CRITICAL", "department": "Emergency",
            "key_missing_info": ["DVT prophylaxis status", "Wells score", "D-dimer",
                                  "lower limb Doppler", "anticoagulation contraindications",
                                  "SpO2 trend", "CTPA availability"],
            "min_missing_to_flag": 3,
            "trap": "Post-op + pleuritic chest pain + tachycardia + SpO2 91 = PE until proven otherwise"
        }
    },
]

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

URGENCY_LEVELS = ["NON_URGENT", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
URGENCY_RANK = {level: idx for idx, level in enumerate(URGENCY_LEVELS)}

DEPARTMENT_KEYWORDS: Dict[str, List[str]] = {
    "Emergency": ["emergency", "er", "ed", "acute", "trauma", "resus"],
    "Cardiology": ["cardiology", "cardiac", "heart", "cardio", "cath"],
    "Pulmonology": ["pulmonology", "pulmonary", "respiratory", "lung"],
    "Orthopedics": ["orthopedics", "ortho", "bone", "joint", "fracture"],
    "Neurology": ["neurology", "neuro", "brain", "stroke", "neurosurgery"],
    "Ophthalmology": ["ophthalmology", "eye", "vision", "ocular"],
    "General": ["general", "primary care", "gp", "family medicine"],
    "Obstetrics": ["obstetrics", "ob", "maternity", "labour", "antenatal"],
    "Paediatrics": ["paediatrics", "pediatrics", "paeds", "peds", "child"],
}
