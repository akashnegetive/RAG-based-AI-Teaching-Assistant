import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE = Path(os.getenv("RAG_DATA_DIR", "rag_data")) / "jsons"
OUT = Path("eval/dataset.jsonl")

FILES = [
    "Ridge And Lasso_Regression.json",
    "Bias And Variance .json",
    "Maximum_Subarray_-_Kadane_s_Algorithm_--_Leetcode_53_144P.json",
    "4_Steps_to_Solve_Any_Dynamic_Programming_DP_Problem_144P.json",
    "Post Prunning And Pre Prunning.json",
]

client = OpenAI()

all_rows = []

for filename in FILES:
    path = BASE / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    title = data["chunks"][0]["title"]
    transcript = "\n".join(c.get("text", "") for c in data["chunks"])

    prompt = f"""
You are creating an evaluation dataset for a RAG teaching assistant.

Lecture title:
{title}

Lecture transcript:
{transcript}

Create exactly 5 evaluation questions based ONLY on the transcript above.

Requirements:
1. Every answer/ground_truth must be directly supported by the transcript.
2. Do not use outside knowledge to fill gaps.
3. Mix question types where the transcript supports it:
   - direct factual/conceptual
   - terminology
   - comparison
   - explanation/application
4. Avoid duplicate questions.
5. Keep ground_truth concise but complete.
6. Return ONLY valid JSON:
{{
  "items": [
    {{
      "question": "...",
      "ground_truth": "..."
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    result = json.loads(response.choices[0].message.content)
    for item in result["items"]:
        all_rows.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "lecture": title,
        })

OUT.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in all_rows) + "\n",
    encoding="utf-8",
)

print(f"Created {len(all_rows)} questions in {OUT}")
