from __future__ import annotations

import os
import sys
import json
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

from src.logger import logging
from src.exception import CustomException
from src.utils import (
    load_dataframe_from_mongo,
    llm_agent,
    fig_to_b64,
    analyse_chart,
    parse_json_response,
    empty_analysis,
)

load_dotenv()

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

_OXLO_API_URL = "https://api.oxlo.ai/v1/chat/completions"
_OXLO_API_KEY = os.getenv("OXLO_API_KEY", "")
_OXLO_MODEL   = "mistral-7b"
_CHART_DPI    = 72


class DataPreprocessing:
    """
    Full preprocessing + EDA pipeline.

    Workflow
    --------
    1. get_ai_insights()          → {col: method} — skips LLM if no nulls
    2. preprocess_data(strategy)  → cleaned DataFrame (in-memory only)
    3. generate_eda_report(df)    → {chart_title: image_b64}
                                    No AI calls here — done on-demand via /analyse_chart
    4. analyse_single(b64, title) → AI analysis dict for one chart
    """

    def __init__(
        self,
        filename: str,
        target_column: str,
        columns_to_drop: list[str] | None = None,
        oxlo_api_key: str = "",
    ) -> None:
        try:
            self.filename        = filename
            self.target_column   = target_column
            self.columns_to_drop = columns_to_drop or []
            self._oxlo_key       = oxlo_api_key or _OXLO_API_KEY

            client  = MongoClient("mongodb://localhost:27017/")
            self.db = client["clarityAI_database"]
            self.fs = gridfs.GridFS(self.db)

            self.df = load_dataframe_from_mongo(filename)
            logging.info("DataPreprocessing loaded '%s' — shape %s", filename, self.df.shape)

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── helpers ───────────────────────────────────────────────────────────────
    def _valid_hue(self, df: pd.DataFrame) -> str | None:
        tc = self.target_column
        if isinstance(tc, str) and tc.strip() and tc in df.columns.tolist():
            return tc
        return None

    # ── 1. AI missing-value strategy ──────────────────────────────────────────
    def get_ai_insights(self) -> dict[str, str]:
        try:
            df = self.df
            null_counts     = df.isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0].to_dict()

            if not cols_with_nulls:
                logging.info("No missing values — skipping AI insights.")
                return {}

            analysis = {
                "shape":               df.shape,
                "null_values":         cols_with_nulls,
            }

            prompt = textwrap.dedent("""
                Given the null_values dict, suggest the SINGLE best method to fill each column.
                Valid methods: mean, median, mode, ffill, bfill, zero, drop

                Return ONLY JSON (no explanation):
                {"column_name": "method", ...}
            """).strip()

            llm_result   = llm_agent(prompt=prompt, role="Data Analyst",
                                     context=json.dumps(analysis, separators=(',', ':')))
            raw_response = llm_result.get("response", "")

            if isinstance(raw_response, dict):
                insights = raw_response
            else:
                insights = parse_json_response(str(raw_response))

            logging.info("AI missing-value strategy: %s", insights)
            return insights

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── 2. Preprocessing ──────────────────────────────────────────────────────
    def preprocess_data(
        self,
        missing_value_strategy: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        try:
            df = self.df.copy()
            logging.info("Preprocessing started.")

            before = len(df)
            df.drop_duplicates(inplace=True)
            logging.info("Duplicates removed: %d rows dropped.", before - len(df))

            if self.columns_to_drop:
                df.drop(
                    columns=[c for c in self.columns_to_drop if c in df.columns],
                    inplace=True,
                )
                logging.info("Dropped columns: %s", self.columns_to_drop)

            if df.isnull().values.any():
                strategy = missing_value_strategy or {}
                for col, method in strategy.items():
                    if col not in df.columns:
                        continue
                    if method in ("mean", "median", "zero"):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    if method == "mean":
                        fill_val = df[col].mean()
                        df[col] = df[col].fillna(fill_val if pd.notna(fill_val) else 0)
                    elif method == "median":
                        fill_val = df[col].median()
                        df[col] = df[col].fillna(fill_val if pd.notna(fill_val) else 0)
                    elif method == "mode":
                        mode_vals = df[col].mode()
                        if not mode_vals.empty:
                            df[col] = df[col].fillna(mode_vals.iloc[0])
                    elif method == "ffill":
                        df[col] = df[col].ffill()
                    elif method == "bfill":
                        df[col] = df[col].bfill()
                    elif method == "zero":
                        df[col] = df[col].fillna(0)
                    elif method == "drop":
                        df.dropna(subset=[col], inplace=True)
                    else:
                        df[col] = df[col].fillna(method)
                logging.info("Missing values handled.")
            else:
                logging.info("No missing values — skipping fill step.")

            logging.info("Preprocessing complete. Final shape: %s", df.shape)
            return df

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── 3. EDA report — charts only, no AI calls ───────────────────────────────
    def generate_eda_report(self, df: pd.DataFrame) -> dict[str, str]:
        """
        Build all charts and return {title: image_b64}.
        AI analysis is NOT done here — it happens on-demand via /analyse_chart.
        Page loads instantly.
        """
        try:
            charts = self._build_all_charts(df)
            if charts is None:
                logging.warning("_build_all_charts returned None — using empty dict.")
                charts = {}
            logging.info("EDA report built — %d charts (no AI calls).", len(charts))
            return charts
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── 4. Analyse a single chart on demand ────────────────────────────────────
    def analyse_single(self, image_b64: str, chart_title: str) -> dict[str, Any]:
        """Called by /analyse_chart route for one chart at a time."""
        try:
            result = analyse_chart(
                image_b64, chart_title,
                self._oxlo_key, _OXLO_API_URL, _OXLO_MODEL,
            )
            logging.info("On-demand analysis done for '%s'.", chart_title)
            return result
        except Exception as e:
            logging.warning("On-demand analysis failed for '%s': %s", chart_title, e)
            return empty_analysis(chart_title)

    # ── Chart builder ──────────────────────────────────────────────────────────
    def _build_all_charts(self, df: pd.DataFrame) -> dict[str, str]:
        charts: dict[str, str] = {}
        try:
            numeric_cols     = df.select_dtypes(include="number").columns.tolist()
            categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
        except Exception as e:
            logging.warning("Could not determine column types: %s", e)
            return charts

        # ── Univariate ─────────────────────────────────────────────────────
        for col in numeric_cols:
            try:
                title = f"Distribution — {col}"
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
                ax.set_title(title); ax.set_xlabel(col)
                charts[title] = fig_to_b64(fig, _CHART_DPI)
            except Exception as e:
                logging.warning("Skipped '%s': %s", title, e)
            finally:
                plt.close("all")

        for col in numeric_cols:
            try:
                title = f"Boxplot — {col}"
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(y=df[col].dropna(), ax=ax, color="#55A868")
                ax.set_title(title)
                charts[title] = fig_to_b64(fig, _CHART_DPI)
            except Exception as e:
                logging.warning("Skipped '%s': %s", title, e)
            finally:
                plt.close("all")

        for col in categorical_cols[:6]:
            try:
                title = f"Value Counts — {col}"
                vc = df[col].value_counts().head(15)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=vc.index, y=vc.values, ax=ax,
                            hue=vc.index, palette="muted", legend=False)
                ax.set_title(title); ax.set_xlabel(col); ax.set_ylabel("Count")
                plt.xticks(rotation=35, ha="right"); plt.tight_layout()
                charts[title] = fig_to_b64(fig, _CHART_DPI)
            except Exception as e:
                logging.warning("Skipped '%s': %s", title, e)
            finally:
                plt.close("all")

        # ── Bivariate ──────────────────────────────────────────────────────
        if len(numeric_cols) >= 2:
            try:
                title = "Correlation Heatmap"
                corr  = df[numeric_cols].corr()
                n     = len(numeric_cols)
                fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                            linewidths=0.5, ax=ax, square=True)
                ax.set_title(title); plt.tight_layout()
                charts[title] = fig_to_b64(fig, _CHART_DPI)
            except Exception as e:
                logging.warning("Skipped 'Correlation Heatmap': %s", e)
            finally:
                plt.close("all")

        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().abs()
            pairs = (
                corr_matrix
                .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                .stack().sort_values(ascending=False).head(3)
            )
            for (col_x, col_y), _ in pairs.items():
                try:
                    title = f"Scatter — {col_x} vs {col_y}"
                    hue   = self._valid_hue(df)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.scatterplot(data=df, x=col_x, y=col_y,
                                    hue=hue, alpha=0.65, ax=ax)
                    ax.set_title(title); plt.tight_layout()
                    charts[title] = fig_to_b64(fig, _CHART_DPI)
                except Exception as e:
                    logging.warning("Skipped '%s': %s", title, e)
                finally:
                    plt.close("all")

        if categorical_cols and numeric_cols:
            try:
                cat_col, num_col = categorical_cols[0], numeric_cols[0]
                title = f"Grouped Bar — {cat_col} × {num_col}"
                grp   = (df.groupby(cat_col)[num_col]
                           .mean().sort_values(ascending=False).head(12))
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=grp.index, y=grp.values, ax=ax,
                            hue=grp.index, palette="Blues_d", legend=False)
                ax.set_title(title); ax.set_xlabel(cat_col)
                ax.set_ylabel(f"Mean {num_col}")
                plt.xticks(rotation=35, ha="right"); plt.tight_layout()
                charts[title] = fig_to_b64(fig, _CHART_DPI)
            except Exception as e:
                logging.warning("Skipped grouped bar: %s", e)
            finally:
                plt.close("all")

        # ── Trivariate ─────────────────────────────────────────────────────
        if len(numeric_cols) >= 3:
            try:
                title     = "Pairplot"
                pair_cols = numeric_cols[:5]
                plot_df   = df[pair_cols].select_dtypes(include="number").dropna()
                if not plot_df.empty:
                    pplot = sns.pairplot(plot_df, hue=None, diag_kind="kde",
                                         plot_kws={"alpha": 0.5})
                    pplot.fig.suptitle(title, y=1.02)
                    charts[title] = fig_to_b64(pplot.fig, _CHART_DPI)
            except Exception as e:
                logging.warning("Skipped 'Pairplot': %s", e)
            finally:
                plt.close("all")

        if len(numeric_cols) >= 3:
            try:
                col_x, col_y, col_s = numeric_cols[0], numeric_cols[1], numeric_cols[2]
                title  = f"Bubble — {col_x} / {col_y} / {col_s}"
                sizes  = pd.to_numeric(df[col_s], errors="coerce").fillna(0)
                scaled = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9) * 400 + 20
                fig, ax = plt.subplots(figsize=(8, 6))
                sc = ax.scatter(df[col_x], df[col_y], s=scaled,
                                alpha=0.5, c=scaled, cmap="viridis")
                plt.colorbar(sc, ax=ax, label=col_s)
                ax.set_xlabel(col_x); ax.set_ylabel(col_y); ax.set_title(title)
                plt.tight_layout()
                charts[title] = fig_to_b64(fig, _CHART_DPI)
            except Exception as e:
                logging.warning("Skipped bubble chart: %s", e)
            finally:
                plt.close("all")

        return charts