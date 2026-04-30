import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

IBM_IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

IBM_API_KEY = os.getenv("IBM_API_KEY")
IBM_AGENT_ID = os.getenv("IBM_AGENT_ID")
IBM_HOST_URL = os.getenv("IBM_HOST_URL", "https://eu-de.watson-orchestrate.cloud.ibm.com")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

app = FastAPI(title="Stock Analysis Agent Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == "*" else [x.strip() for x in ALLOWED_ORIGINS.split(",") if x.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1)

REQUIRED_SCORE_KEYS = [
    "Momentum",
    "Volatility",
    "News Sentiment",
    "Fundamentals",
    "Investor Fit",
]

def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    safe = dict(headers)
    auth = safe.get("Authorization")
    if auth:
        if auth.startswith("Bearer "):
            token = auth[len("Bearer "):]
            safe["Authorization"] = f"Bearer {token[:8]}...{token[-8:]}" if len(token) > 16 else "Bearer <redacted>"
        else:
            safe["Authorization"] = "<redacted>"
    return safe

def clamp_score(value: Any) -> int:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        n = 1
    return max(1, min(5, n))

def extract_json_from_text(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    fenced_any = re.search(r"```\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced_any:
        try:
            parsed = json.loads(fenced_any.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None

def normalize_payload(obj: dict[str, Any]) -> dict[str, Any]:
    scores = obj.get("scores") if isinstance(obj.get("scores"), dict) else {}
    normalized_scores = {k: clamp_score(scores.get(k, 1)) for k in REQUIRED_SCORE_KEYS}

    recommendation = str(obj.get("recommendation", "Watchlist")).strip()
    if recommendation not in {"Buy", "Watchlist", "Avoid"}:
        recommendation = "Watchlist"

    confidence = str(obj.get("confidence", "Medium")).strip()
    if confidence not in {"High", "Medium", "Low"}:
        confidence = "Medium"

    risk_note = obj.get("risk_note")
    if not isinstance(risk_note, list):
        risk_note = []
    risk_note = [str(x).strip() for x in risk_note if str(x).strip()][:2]

    rationale = str(obj.get("rationale", "")).strip() or "The agent returned an incomplete rationale."
    disclaimer = str(
        obj.get(
            "disclaimer",
            "This analysis is for educational purposes only and not financial advice."
        )
    ).strip()

    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "scores": normalized_scores,
        "rationale": rationale,
        "risk_note": risk_note,
        "disclaimer": disclaimer,
    }

def get_iam_access_token() -> str:
    if not IBM_API_KEY:
        raise HTTPException(status_code=500, detail="Missing IBM_API_KEY environment variable")

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": IBM_API_KEY,
    }

    print("\n=== IAM TOKEN REQUEST ===")
    print("URL:", IBM_IAM_TOKEN_URL)
    print("Headers:", headers)

    try:
        response = requests.post(IBM_IAM_TOKEN_URL, headers=headers, data=data, timeout=30)
        print("Status:", response.status_code)
        print("Body:", response.text[:2000])
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"IAM token request failed: {exc}") from exc

    token = payload.get("access_token")
    if not token:
        raise HTTPException(status_code=502, detail=f"IAM token response missing access_token: {payload}")

    return token

def extract_text_from_agent_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        elif isinstance(item.get("content"), str):
                            parts.append(item["content"])
                if parts:
                    return "\n".join(parts)
        if isinstance(first.get("text"), str):
            return first["text"]

    if isinstance(payload.get("content"), str):
        return payload["content"]

    return json.dumps(payload)

def call_ibm_agent(query: str) -> dict[str, Any]:
    if not IBM_AGENT_ID:
        raise HTTPException(status_code=500, detail="Missing IBM_AGENT_ID environment variable")
    if not IBM_HOST_URL:
        raise HTTPException(status_code=500, detail="Missing IBM_HOST_URL environment variable")

    token = get_iam_access_token()
    url = f"{IBM_HOST_URL}/api/v1/orchestrate/{IBM_AGENT_ID}/chat/completions"

    body = {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    print("\n=== AGENT REQUEST ===")
    print("URL:", url)
    print("Headers:", redact_headers(headers))
    print("Body:", json.dumps(body, indent=2))

    try:
        response = requests.post(url, headers=headers, json=body, timeout=120)
        print("Status:", response.status_code)
        print("Body:", response.text[:4000])
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        try:
            detail = f" | Response: {response.text[:4000]}"
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=f"Agent call failed: {exc}{detail}") from exc

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    agent_payload = call_ibm_agent(req.query)
    agent_text = extract_text_from_agent_response(agent_payload)
    parsed = extract_json_from_text(agent_text)

    if not parsed:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Agent response was not valid JSON",
                "raw_agent_text": agent_text,
            },
        )

    return normalize_payload(parsed)