import json
import logging
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IBM_IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

IBM_API_KEY = os.getenv("IBM_API_KEY", "").strip()
IBM_AGENT_ID = os.getenv("IBM_AGENT_ID", "").strip()
IBM_API_ENDPOINT = os.getenv("IBM_API_ENDPOINT", "").strip().rstrip("/")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").strip()

app = FastAPI(title="Stock Analysis Agent Backend", version="1.0.0")


def parse_allowed_origins(origins_raw: str) -> list[str]:
    if not origins_raw or origins_raw == "*":
        return ["*"]
    return [origin.strip() for origin in origins_raw.split(",") if origin.strip()]


allowed_origins = parse_allowed_origins(ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False if allowed_origins == ["*"] else True,
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


def clamp_score(value: Any) -> int:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        n = 0
    return max(0, min(5, n))


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    safe = dict(headers)
    auth = safe.get("Authorization")
    if auth:
        if auth.startswith("Bearer "):
            token = auth[len("Bearer "):]
            safe["Authorization"] = (
                f"Bearer {token[:8]}...{token[-8:]}" if len(token) > 16 else "Bearer <redacted>"
            )
        else:
            safe["Authorization"] = "<redacted>"
    return safe


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    text = text.strip()

    # 1. Direct JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. JSON fenced block
    fenced_json = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced_json:
        candidate = fenced_json.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Any fenced block
    fenced_any = re.search(r"```\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced_any:
        candidate = fenced_any.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 4. First JSON-looking object in text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidate = match.group(0).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def normalize_payload(obj: dict[str, Any]) -> dict[str, Any]:
    scores = obj.get("scores") if isinstance(obj.get("scores"), dict) else {}
    normalized_scores = {key: clamp_score(scores.get(key, 0)) for key in REQUIRED_SCORE_KEYS}

    recommendation = str(obj.get("recommendation", "Watchlist")).strip()
    if recommendation not in {"Buy", "Watchlist", "Avoid"}:
        recommendation = "Cannot determine"

    confidence = str(obj.get("confidence", "Medium")).strip()
    if confidence not in {"High", "Medium", "Low"}:
        confidence = "Cannot determine"

    risk_note = obj.get("risk_note")
    if not isinstance(risk_note, list):
        risk_note = []
    risk_note = [str(x).strip() for x in risk_note if str(x).strip()][:2]

    rationale = str(obj.get("rationale", "")).strip()
    if not rationale:
        rationale = "The agent returned an incomplete rationale."

    disclaimer = str(
        obj.get(
            "disclaimer",
            "This analysis is for educational purposes only and not financial advice.",
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

    response = None
    try:
        response = requests.post(IBM_IAM_TOKEN_URL, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        detail = ""
        if response is not None:
            detail = f" | Response: {response.text[:2000]}"
        raise HTTPException(status_code=502, detail=f"IAM token request failed: {exc}{detail}") from exc

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
    if not IBM_API_ENDPOINT:
        raise HTTPException(status_code=500, detail="Missing IBM_API_ENDPOINT environment variable")

    token = get_iam_access_token()
    url = f"{IBM_API_ENDPOINT}/v1/orchestrate/{IBM_AGENT_ID}/chat/completions"
    body = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ],
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    logger.info("Sending request to IBM agent")
    logger.info("URL: %s", url)
    logger.info("Headers: %s", redact_headers(headers))
    logger.info("Body: %s", json.dumps(body))

    response = None
    try:
        response = requests.post(url, headers=headers, json=body, timeout=120)
        logger.info("Agent status: %s", response.status_code)
        logger.info("Agent response preview: %s", response.text[:4000])
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        if response is not None:
            detail = f" | Response: {response.text[:4000]}"
        raise HTTPException(status_code=502, detail=f"Agent call failed: {exc}{detail}") from exc


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
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