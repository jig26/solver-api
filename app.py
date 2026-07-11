from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
Solve this arithmetic word problem.

Rules:
- Think carefully.
- Ignore irrelevant numbers.
- Return JSON ONLY.
- reasoning must be at least 80 characters.
- answer must be an INTEGER.

Problem:
{req.problem}
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    import json

    text = response.output_text
    result = json.loads(text)

    return {
        "reasoning": str(result["reasoning"]),
        "answer": int(result["answer"])
    }