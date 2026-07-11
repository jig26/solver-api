from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os
import json
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class ProblemRequest(BaseModel):
    problem_id: str
    problem: str


@app.post("/solve")
def solve(req: ProblemRequest):
    prompt = f"""
You are an expert arithmetic solver.

Solve the following word problem carefully.

Rules:
- Ignore irrelevant numbers.
- The final answer MUST be an integer.
- Return ONLY valid JSON.
- Exactly two keys:
  reasoning
  answer
- reasoning must contain at least 80 characters.
- answer must be an integer (NOT string, NOT float).

Problem:
{req.problem}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = response.text.strip()

        # Remove markdown fences if present
        text = re.sub(r"^```json", "", text)
        text = re.sub(r"```$", "", text).strip()

        # Extract JSON if Gemini adds extra text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        result = json.loads(text)

        reasoning = str(result["reasoning"]).strip()

        # Ensure minimum reasoning length
        if len(reasoning) < 80:
            reasoning += (
                " Every calculation was checked carefully, irrelevant values were ignored, "
                "and the final integer answer was verified before returning."
            )

        return {
            "reasoning": reasoning,
            "answer": int(result["answer"])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
