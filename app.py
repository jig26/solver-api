from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import re
import ollama

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Problem(BaseModel):
    problem_id: str
    problem: str


@app.post("/solve")
def solve(req: Problem):

    prompt = f"""
You are solving arithmetic word problems.

Rules:
- Ignore irrelevant numbers.
- Think carefully.
- Return ONLY valid JSON.
- Exactly two keys:
  reasoning
  answer
- reasoning must be at least 80 characters.
- answer must be an integer.

Problem:
{req.problem}
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        text = response["message"]["content"].strip()

        # extract JSON if wrapped in extra text
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group()

        result = json.loads(text)

        return {
            "reasoning": str(result["reasoning"]),
            "answer": int(result["answer"])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
