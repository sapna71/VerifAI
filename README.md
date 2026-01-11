# VerifAI
# Kharagpur Data Science Hackathon 2026  
## Story–Backstory Consistency Detection

### Team Submission – KDSH 2026

---

## 📌 Problem Overview

This project addresses the **Story–Backstory Consistency Detection** task from the **Kharagpur Data Science Hackathon 2026**.

Given:
- A **story** (narrative text)
- A **backstory** (character traits / background claims)

The goal is to automatically determine whether the backstory is:
- **Consistent** with the story  
- **Contradictory** to the story  

The system outputs:
- A binary prediction (0 = contradiction, 1 = consistent)
- A confidence score
- A short natural-language rationale

---

## 🧠 Methodology

Our approach follows a **retrieve-and-judge pipeline**:

### 1️⃣ Claim Extraction
- The backstory text is split into atomic claims using sentence-level segmentation.
- Each claim is evaluated independently.

### 2️⃣ Evidence Retrieval
- The story is split into overlapping text chunks.
- Sentence embeddings are generated using **Sentence Transformers**.
- For each claim, the top-k most relevant story chunks are retrieved via cosine similarity.

### 3️⃣ Consistency Judgement
- Retrieved evidence + claim are passed to a **Natural Language Inference (NLI)** model.
- We use **RoBERTa-large-MNLI** to classify each claim as:
  - entailment (consistent)
  - contradiction
  - neutral

### 4️⃣ Aggregation
- Claim-level predictions are aggregated to produce a final story-level label.
- Confidence scores are averaged across claims.

---

## 🏗️ Project Structure

kdsh_2026/

│
├── main.py # Entry point
├── requirements.txt
├── README.md
│
├── data/
│ ├── story.txt
│ ├── backstory.txt
│ └── dataset.csv # (optional, for evaluation)
│
└── src/

├── pipeline.py # Orchestrates the full pipeline
├── retrieve.py # Semantic retrieval
├── judge.py # NLI-based consistency checking
└── io.py # Data loading, saving, accuracy computation



---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Run inference
python main.py

3️⃣ Output

Example output:

Prediction: 1
Rationale: Claim is consistent with the story (confidence: 0.76)

📊 Evaluation & Accuracy

Accuracy is computed when ground-truth labels are available in dataset.csv.

The system compares predicted labels against true labels and reports:

Accuracy score

Evaluation logic is implemented in src/io.py.

⚠️ Due to the open-ended nature of narrative text, performance depends on:

Story length

Backstory granularity

Claim ambiguity

📦 Models Used
Component	Model
Embeddings	sentence-transformers/all-MiniLM-L6-v2
NLI	roberta-large-mnli
🧪 Example

Story:

Once upon a time, there was a curious little girl named Alice who loved exploring...

Backstory:

The character grew up in poverty and believes violence is justified for survival.

Prediction:

Contradiction (confidence: 0.76)

⚠️ Notes & Limitations

Long stories are chunked to avoid transformer length limits.

Neutral NLI outputs are treated as weak contradiction.

CPU inference may be slow for large inputs.

👨‍💻 Team & Submission

Hackathon: Kharagpur Data Science Hackathon 2026

Track: NLP / Consistency Detection

Submission Type: Code + Model-based Approach




