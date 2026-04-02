import sys
import io
import os
import json
import re
import base64
import textwrap
import time
from typing import Any

import pandas as pd
import gridfs
from pymongo import MongoClient
import openai

from src.logger import logging
from src.exception import CustomException


# ── MongoDB / GridFS ───────────────────────────────────────────────────────────

def get_gridfs_connection():
    """Create and return MongoDB GridFS connection."""
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["clarityAI_database"]
        fs = gridfs.GridFS(db)
        logging.info("MongoDB GridFS connection established.")
        return fs
    except Exception as e:
        raise CustomException(e, sys)


def fetch_csv_from_gridfs(filename: str):
    """Fetch CSV file bytes from GridFS using filename."""
    try:
        fs = get_gridfs_connection()
        grid_out = fs.find_one(
            {"filename": filename},
            sort=[("uploadDate", -1)]
        )
        if grid_out is None:
            raise Exception(f"File '{filename}' not found in GridFS")
        logging.info("CSV file '%s' fetched from GridFS.", filename)
        return grid_out.read()
    except Exception as e:
        raise CustomException(e, sys)


def load_dataframe_from_mongo(filename: str) -> pd.DataFrame:
    """Load CSV from MongoDB GridFS and return a pandas DataFrame."""
    try:
        file_bytes = fetch_csv_from_gridfs(filename)
        df = pd.read_csv(io.BytesIO(file_bytes))
        logging.info("Dataset successfully loaded into DataFrame.")
        return df
    except Exception as e:
        raise CustomException(e, sys)


# ── LLM (OXLO) ────────────────────────────────────────────────────────────────

def llm_agent(prompt: str, role: str, context: str) -> dict:
    """
    Send a structured prompt to OXLO and return the parsed JSON response.
    Includes retry logic with exponential backoff for rate limits (429).

    Returns
    -------
    dict with keys:
        "response"  → the LLM's answer string
        "metadata"  → role, original_query, context_summary
    """
    max_retries = 5
    retry_delay = 1  # Start with 1 second, exponential backoff: 1s, 2s, 4s, 8s
    
    for attempt in range(max_retries):
        try:
            client = openai.OpenAI(
                base_url="https://api.oxlo.ai/v1",
                api_key=os.environ.get("OXLO_API_KEY")
            )

            structured_prompt = (
                f"Act as {role} and respond only in JSON format.\n"
                f"The context is: {context}\n"
                f"The user query is: {prompt}\n\n"
                "The JSON structure must always be:\n"
                "{\n"
                '    "response": "<your answer here>",\n'
                '    "metadata": {\n'
                '        "role": "<role>",\n'
                '        "original_query": "<original user query>",\n'
                '        "context_summary": "<short summary of context>"\n'
                "    }\n"
                "}"
            )

            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": structured_prompt}],
                model="llama-3.2-3b",
                max_tokens=4096,
                temperature=0.3,
            )

            raw_content = chat_completion.choices[0].message.content

            try:
                # Try to parse JSON response
                parsed = json.loads(raw_content)
                return parsed
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract valid JSON from the response
                start_idx = raw_content.find('{')
                end_idx = raw_content.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    try:
                        parsed = json.loads(raw_content[start_idx:end_idx+1])
                        return parsed
                    except json.JSONDecodeError:
                        pass
                
                # If all JSON parsing fails, return wrapped response
                return {
                    "response": raw_content,
                    "metadata": {
                        "role": role,
                        "original_query": prompt,
                        "context_summary": context[:100],
                    }
                }

        except openai.RateLimitError as e:
            # Handle 429 rate limit error with exponential backoff
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                logging.warning(f"Rate limited (429). Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"LLM Agent failed after {max_retries} retries: {e}")
        
        except Exception as e:
            raise Exception(f"LLM Agent failed: {e}")


# ── Chart utilities ────────────────────────────────────────────────────────────

def fig_to_b64(fig, dpi: int = 72) -> str:
    """
    Render a matplotlib Figure to a base64-encoded PNG string.
    No disk I/O — uses BytesIO only.
    """
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def analyse_chart(
    image_b64: str,
    chart_title: str,
    oxlo_api_key: str,
    oxlo_api_url: str,
    oxlo_model: str,
) -> dict[str, Any]:
    """
    Send a base64 PNG to the Oxlo/Mistral vision endpoint via OpenAI client.
    Returns a structured JSON dict with keys:
        represents, key_findings, anomalies, recommendations
    Thread-safe — no shared state.
    """
    import openai

    system_prompt = textwrap.dedent("""
        You are a senior data analyst reviewing a chart image.
        Return ONLY a valid JSON object — no preamble, no markdown fences.
        Schema:
        {
          "represents":      "<one sentence: what this chart shows>",
          "key_findings":    ["<finding 1>", "<finding 2>", ...],
          "anomalies":       ["<anomaly 1>", ...],
          "recommendations": ["<action 1>", ...]
        }
    """).strip()

    base_url = oxlo_api_url.replace("/chat/completions", "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
            {
                "type": "text",
                "text": f"Chart title: {chart_title}\n\nAnalyse and return JSON.",
            },
        ]},
    ]

    # Retry up to 2 times on failure
    for attempt in range(1, 3):
        try:
            client = openai.OpenAI(
                base_url=base_url,
                api_key=oxlo_api_key,
                timeout=90,
            )
            response = client.chat.completions.create(
                model=oxlo_model,
                max_tokens=1024,
                temperature=0.3,
                messages=messages,
            )
            raw = response.choices[0].message.content.strip()
            logging.info("Oxlo analysis received for '%s' (attempt %d).", chart_title, attempt)
            return parse_json_response(raw)

        except Exception as exc:
            logging.warning(
                "Oxlo API attempt %d failed for '%s': %s",
                attempt, chart_title, exc,
            )
            if attempt == 2:
                return empty_analysis(chart_title)


def parse_json_response(raw: str) -> dict:
    """
    Parse JSON from an LLM response with robust fallbacks.
    Handles markdown fences, control characters, and malformed JSON.
    """
    raw = raw.strip()
    
    # Remove control characters
    raw = raw.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Strip markdown code blocks
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.strip().lower().startswith("json"):
            raw = raw.strip()[4:]
    
    raw = raw.strip()
    
    # Try direct JSON parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Try to extract valid JSON substring
        start_idx = raw.find('{')
        end_idx = raw.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                extracted = raw[start_idx:end_idx+1]
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass
        
        # Fallback: regex search for JSON object (non-greedy)
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        logging.warning("JSON parse failed at pos %d: %s. Response: %s", e.pos, e.msg, raw[:300])
        return {}


def empty_analysis(title: str) -> dict[str, Any]:
    """Return a blank analysis stub used when AI analysis is skipped or fails."""
    return {
        "represents":      title,
        "key_findings":    [],
        "anomalies":       [],
        "recommendations": [],
    }