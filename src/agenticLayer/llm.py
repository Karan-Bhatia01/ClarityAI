from __future__ import annotations

import json
import os
import sys
from typing import Any

import openai
from dotenv import load_dotenv

from src.logger import logging
from src.exception import CustomException
from src.utils import load_dataframe_from_mongo

load_dotenv()

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
_LLM_MODEL = "llama-3.2-3b"
_LLM_MAX_TOKENS = 4_096
_UNIQUE_PREVIEW_LIMIT = 10

_SYSTEM_PROMPT = """\
You are a data analyst. Analyze the provided dataset and respond with ONLY a valid JSON object.
NO markdown, NO code blocks, NO preamble. Just the JSON.
Include all these keys: summary, quality_flags, column_insights, next_steps, uncertainty_notes
"""


# ──────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────
class AnalysisExplainer:
    """
    Loads a dataset from MongoDB GridFS, computes descriptive statistics
    using pandas, and generates structured JSON insights via OXLO LLM.

    Usage
    -----
    explainer = AnalysisExplainer("my_dataset.csv")
    result = explainer.run()
    # result["analysis"]    → dict of computed statistics
    # result["unique"]      → unique value previews dict
    # result["ai_insights"] → structured dict (summary, quality_flags, etc.)
    """

    def __init__(self, filename: str) -> None:
        try:
            self.filename = filename
            self.df = self._load_dataframe()
            self.client = openai.OpenAI(
                base_url="https://api.oxlo.ai/v1",
                api_key=self._require_env("OXLO_API_KEY")
            )
            logging.info("AnalysisExplainer initialised for file: %s", filename)
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── private helpers ────────────────────────

    @staticmethod
    def _require_env(key: str) -> str:
        """Return env variable *key* or raise a clear error if missing."""
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(
                f"Required environment variable '{key}' is not set."
            )
        return value

    def _load_dataframe(self):
        """Load dataset from MongoDB GridFS via utility helper."""
        try:
            df = load_dataframe_from_mongo(self.filename)
            logging.info("DataFrame loaded: %d rows × %d columns.", *df.shape)
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── public methods ─────────────────────────

    def compute_analysis(self) -> dict[str, Any]:
        """
        Compute descriptive statistics using pandas only.
        The LLM never performs calculations — all numbers originate here.

        Returns
        -------
        dict with keys:
            shape, columns, dtypes, null_values, null_percentages,
            duplicate_rows, numeric_columns, categorical_columns,
            describe, memory_usage_mb, unique_counts, sample_rows
        """
        try:
            df = self.df
            total_rows = len(df)

            null_counts: dict[str, int] = df.isnull().sum().to_dict()
            null_pct: dict[str, float] = {
                col: round(count / total_rows * 100, 2)
                for col, count in null_counts.items()
            }

            analysis: dict[str, Any] = {
                "shape": df.shape,
                "memory_usage_mb": round(
                    df.memory_usage(deep=True).sum() / 1024 ** 2, 3
                ),
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
                "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
                "null_values": null_counts,
                "null_percentages": null_pct,
                "duplicate_rows": int(df.duplicated().sum()),
                "describe": df.describe(include="all").to_dict(),
                "unique_counts": df.nunique().to_dict(),
                "sample_rows": df.head(5).to_dict(orient="records"),
            }

            logging.info("Dataset analysis computed successfully.")
            return analysis

        except Exception as e:
            raise CustomException(e, sys) from e

    def unique_preview(
        self, limit: int = _UNIQUE_PREVIEW_LIMIT
    ) -> dict[str, dict[str, Any]]:
        """
        Return a preview of unique values for every column.

        Returns
        -------
        dict: column → {"values": [...], "total_unique": int, "truncated": bool}
        """
        try:
            preview: dict[str, dict[str, Any]] = {}
            for col in self.df.columns:
                unique_vals = self.df[col].dropna().unique()
                total = len(unique_vals)
                preview[col] = {
                    "values": unique_vals[:limit].tolist(),
                    "total_unique": total,
                    "truncated": total > limit,
                }
            logging.info("Unique value preview computed.")
            return preview
        except Exception as e:
            raise CustomException(e, sys) from e

    def explain_analysis(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """
        Send pre-computed statistics to the LLM with AGGRESSIVE payload compression.
        
        Strategy: Multiple fallback levels to ensure payload stays under 50KB hard limit.
        """
        try:
            # LEVEL 0: Filter & select only essential information
            essential_keys = {"shape", "columns", "null_values", "null_percentages"}
            prompt_data = {k: v for k, v in analysis.items() if k in essential_keys}

            # Add LIMITED describe stats (only 5 columns max, only basic stats)
            if "describe" in analysis:
                describe_dict = analysis["describe"]
                cols_to_include = list(describe_dict.keys())[:5]
                
                prompt_data["describe_sample"] = {
                    col: {
                        stat: round(val, 2) if isinstance(val, float) else val
                        for stat, val in describe_dict[col].items()
                        if stat in ("count", "mean", "std")
                    }
                    for col in cols_to_include
                }

            # Build COMPACT message (single-line JSON, no indentation)
            user_message = (
                "Analyze this dataset summary and respond ONLY with valid JSON:\n"
                f"{json.dumps(prompt_data, separators=(',', ':'), default=str)}\n\n"
                "REQUIRED JSON structure:\n"
                '{"summary":"<2-3 sentences analyzing the dataset>","quality_flags":[{"column":"name","severity":"high/medium/low","issue":"label","detail":"explanation"}],"column_insights":[{"column":"name","insight":"interpretation"}],"next_steps":[{"title":"action","detail":"explanation"}],"uncertainty_notes":"text"}'
            )
            
            payload_size = len(user_message.encode('utf-8'))
            logging.info("Payload: %d bytes", payload_size)

            # If STILL too large after aggressive reduction, make it even smaller
            if payload_size > 50_000:
                logging.warning("Payload still large (%d bytes). Final reduction.", payload_size)
                # Remove describe entirely
                if "describe_sample" in prompt_data:
                    del prompt_data["describe_sample"]
                # Only keep null info + shape
                prompt_data = {
                    "rows": analysis["shape"][0],
                    "columns": len(analysis["shape"][1]) if isinstance(analysis["shape"], tuple) else analysis["shape"][1],
                    "null_info": {k: v for k, v in analysis.get("null_percentages", {}).items() if v > 0},
                }
                user_message = f"Dataset: {json.dumps(prompt_data, separators=(',', ':'), default=str)}. Analyze briefly."
                payload_size = len(user_message.encode('utf-8'))
                logging.info("Final payload: %d bytes", payload_size)

            completion = self.client.chat.completions.create(
                model=_LLM_MODEL,
                max_tokens=_LLM_MAX_TOKENS,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )

            raw: str = completion.choices[0].message.content.strip()
            logging.debug("Raw LLM response (first 500 chars): %s", raw[:500])

            # Aggressive markdown/code block stripping
            if raw.startswith("```"):
                # Extract content between triple backticks
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.strip().startswith("json"):
                    raw = raw.strip()[4:]
            
            raw = raw.strip()
            
            # Handle common OXLO response issues
            # Remove any control characters or extra whitespace
            raw = raw.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Try standard JSON parsing first
            try:
                insights: dict[str, Any] = json.loads(raw)
                logging.info("Successfully parsed OXLO JSON response")
                
                # Check if response has meaningful insights
                has_flags = bool(insights.get("quality_flags"))
                has_insights = bool(insights.get("column_insights"))
                has_steps = bool(insights.get("next_steps"))
                
                # If all insight fields are empty, use statistical fallback
                if not has_flags and not has_insights and not has_steps:
                    logging.warning("OXLO response has no insights. Using statistical analysis fallback.")
                    return self._build_insights_from_stats(analysis)
                    
            except json.JSONDecodeError:
                # If that fails, try to extract valid JSON from the response
                # Look for the first { and last } and extract that substring
                start_idx = raw.find('{')
                end_idx = raw.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    raw = raw[start_idx:end_idx+1]
                    insights: dict[str, Any] = json.loads(raw)
                    logging.info("Successfully extracted OXLO JSON from text")
                else:
                    logging.warning("Could not find JSON in OXLO response. Using fallback insights.")
                    insights = self._build_insights_from_stats(analysis)
            
            logging.info("AI insights generated successfully.")
            
            # Validate and fix missing fields
            insights = self._validate_insights(insights)
            return insights

        except json.JSONDecodeError as exc:
            logging.warning("JSON parse error at position %d: %s. Building from stats instead.", 
                          exc.pos, exc.msg)
            return self._build_insights_from_stats(analysis)
        except Exception as e:
            logging.error("Unexpected error in explain_analysis: %s", e)
            return self._build_insights_from_stats(analysis)

    def _build_insights_from_stats(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """
        Build insights from statistical data when AI analysis fails.
        Generates meaningful observations from the raw statistics.
        """
        try:
            shape = analysis.get("shape", (0, 0))
            nulls = analysis.get("null_percentages", {})
            duplicates = analysis.get("duplicate_rows", 0)
            
            # Build summary
            summary_parts = [f"Dataset contains {shape[0]} rows and {shape[1]} columns."]
            if duplicates > 0:
                summary_parts.append(f"Found {duplicates} duplicate rows.")
            null_cols = [c for c, p in nulls.items() if p > 0]
            if null_cols:
                summary_parts.append(f"Columns with missing values: {', '.join(null_cols)}.")
            
            # Build quality flags
            quality_flags = []
            for col, pct in nulls.items():
                if pct > 10:
                    quality_flags.append({
                        "column": col,
                        "severity": "high" if pct > 50 else "medium",
                        "issue": "Missing values",
                        "detail": f"{pct}% of values are missing in this column."
                    })
            
            if duplicates > 0:
                quality_flags.append({
                    "column": "dataset",
                    "severity": "medium",
                    "issue": "Duplicate rows",
                    "detail": f"Found {duplicates} duplicate rows that may need investigation."
                })
            
            # Build column insights from describe stats
            column_insights = []
            describe = analysis.get("describe", {})
            numeric_cols = analysis.get("numeric_columns", [])
            
            for col in numeric_cols[:5]:  # Limit to first 5
                if col in describe:
                    stats = describe[col]
                    mean = stats.get("mean", 0)
                    std = stats.get("std", 0)
                    column_insights.append({
                        "column": col,
                        "insight": f"Mean: {mean:.2f}, Std Dev: {std:.2f}. " +
                                 ("High variability detected." if std > mean else "Low variability.")
                    })
            
            # Next steps
            next_steps = [
                {"title": "Review Data Quality", "detail": "Examine null values and duplicates to understand data integrity."},
                {"title": "Prepare for Preprocessing", "detail": "Consider handling missing values and removing duplicates before modeling."},
                {"title": "Explore Relationships", "detail": "Analyze correlations between numeric columns to identify patterns."}
            ]
            
            result = {
                "summary": " ".join(summary_parts),
                "quality_flags": quality_flags,
                "column_insights": column_insights,
                "next_steps": next_steps,
                "uncertainty_notes": "This analysis is based on statistical summaries. AI-powered insights were unavailable.",
            }
            
            return result
        except Exception as e:
            logging.warning("Error building insights from stats: %s", e)
            return self._default_insights()

    def _validate_insights(self, insights: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure all required keys exist and have correct types.
        Provides sensible defaults for missing or malformed fields.
        """
        defaults = {
            "summary": "Dataset analysis completed.",
            "quality_flags": [],
            "column_insights": [],
            "next_steps": [],
            "uncertainty_notes": "Standard statistical limitations apply.",
        }
        
        # Ensure all keys exist
        for key, default_val in defaults.items():
            if key not in insights:
                insights[key] = default_val
            elif insights[key] is None:
                insights[key] = default_val
        
        # Fix types if needed
        if not isinstance(insights["summary"], str):
            insights["summary"] = str(insights["summary"])
        
        if not isinstance(insights["quality_flags"], list):
            insights["quality_flags"] = []
        else:
            # Validate each flag structure
            validated_flags = []
            for flag in insights["quality_flags"]:
                if isinstance(flag, dict) and all(k in flag for k in ["column", "severity", "issue", "detail"]):
                    validated_flags.append(flag)
            insights["quality_flags"] = validated_flags
        
        if not isinstance(insights["column_insights"], list):
            insights["column_insights"] = []
        else:
            # Validate each insight structure
            validated_insights = []
            for insight in insights["column_insights"]:
                if isinstance(insight, dict) and "column" in insight and "insight" in insight:
                    validated_insights.append(insight)
            insights["column_insights"] = validated_insights
        
        if not isinstance(insights["next_steps"], list):
            insights["next_steps"] = []
        else:
            # Validate each step structure
            validated_steps = []
            for step in insights["next_steps"]:
                if isinstance(step, dict) and "title" in step and "detail" in step:
                    validated_steps.append(step)
            insights["next_steps"] = validated_steps
        
        if not isinstance(insights["uncertainty_notes"], str):
            insights["uncertainty_notes"] = str(insights["uncertainty_notes"])
        
        return insights
    
    def _default_insights(self) -> dict[str, Any]:
        """Return a valid default response when parsing completely fails."""
        return {
            "summary": "Dataset analysis completed. Check the raw statistics above for details.",
            "quality_flags": [],
            "column_insights": [],
            "next_steps": [
                {"title": "Review Statistics", "detail": "Examine the analysis results shown above."}
            ],
            "uncertainty_notes": "AI analysis unavailable; refer to standard statistical measures.",
        }

    def run(self) -> dict[str, Any]:
        """
        Execute the full pipeline: load → analyse → explain.

        Returns
        -------
        dict with keys:
            "analysis"     → raw statistics dict
            "unique"       → unique value previews dict
            "ai_insights"  → structured JSON dict from LLM
        """
        try:
            logging.info("Starting analysis pipeline for '%s'.", self.filename)
            analysis = self.compute_analysis()
            unique = self.unique_preview()
            ai_insights = self.explain_analysis(analysis)

            logging.info("Pipeline completed successfully.")
            return {
                "analysis": analysis,
                "unique": unique,
                "ai_insights": ai_insights,
            }

        except Exception as e:
            raise CustomException(e, sys) from e