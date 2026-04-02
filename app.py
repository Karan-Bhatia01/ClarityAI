import os

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    session,
)
from src.components.data_ingestion import DataIngestion
from src.agenticLayer.llm import AnalysisExplainer

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-in-production")
data_ingestion = DataIngestion()


# ── Vera Landing Page (Root) ──────────────────────────────────────────────────
@app.route("/")
def vera_landing():
    return render_template("vera_landing.html")


# ── Upload Page ────────────────────────────────────────────────────────────────
@app.route("/upload", methods=["GET", "POST"])
def index():
    selected_filename = request.args.get("filename")

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            data_ingestion.store_file(file)
            return redirect(url_for("index", filename=file.filename))

    filenames = data_ingestion.get_all_filenames()
    preview_data, columns = data_ingestion.get_preview()

    return render_template(
        "index.html",
        filenames=filenames,
        selected_filename=selected_filename,
        preview_data=preview_data,
        columns=columns,
    )


# ── Dataset Info + AI Insights ─────────────────────────────────────────────────
@app.route("/info")
def info_layer():
    filename = request.args.get("filename")
    if not filename:
        return redirect(url_for("index"))

    ai_explainer = AnalysisExplainer(filename=filename)
    ai_result    = ai_explainer.run()
    print(ai_result)
    return render_template(
        "info.html",
        analysis=ai_result["analysis"],
        unique=ai_result["unique"],
        ai_insights=ai_result["ai_insights"],
    )


# ── Preprocessing + EDA ────────────────────────────────────────────────────────
@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing_inputs():
    filename = request.args.get("filename")
    if not filename:
        return redirect(url_for("index"))

    from src.components.eda_processing import DataPreprocessing

    dp = DataPreprocessing(
        filename=filename,
        target_column="",
        oxlo_api_key=os.environ.get("OXLO_API_KEY", ""),
    )

    if request.method == "GET":
        ai_strategy = dp.get_ai_insights()
        
        # Restore previously submitted form values from session
        previous_target = session.get(f"prev_target_{filename}", "")
        previous_columns_to_drop = session.get(f"prev_cols_drop_{filename}", [])
        previous_missing_strategy = session.get(f"prev_missing_{filename}", {})
        
        return render_template(
            "eda_processing.html",
            filename=filename,
            columns=dp.df.columns.tolist(),
            null_counts=dp.df.isnull().sum().to_dict(),
            ai_strategy=ai_strategy,
            has_nulls=bool(dp.df.isnull().values.any()),
            shape=dp.df.shape,
            # Pass back previous values to re-populate form
            previous_target=previous_target,
            previous_columns_to_drop=previous_columns_to_drop,
            previous_missing_strategy=previous_missing_strategy,
        )

    # POST — preprocess then build charts (no AI calls)
    target_column   = request.form.get("target_column", "")
    columns_to_drop = request.form.getlist("columns_to_drop")

    missing_value_strategy = {}
    for col in dp.df.columns:
        strategy = request.form.get(f"missing_{col}")
        if strategy:
            missing_value_strategy[col] = strategy

    # Save form values to session for later retrieval
    session[f"prev_target_{filename}"] = target_column
    session[f"prev_cols_drop_{filename}"] = columns_to_drop
    session[f"prev_missing_{filename}"] = missing_value_strategy
    session.modified = True

    dp.target_column   = target_column
    dp.columns_to_drop = columns_to_drop

    cleaned_df = dp.preprocess_data(missing_value_strategy=missing_value_strategy)
    eda_report = dp.generate_eda_report(cleaned_df)   # returns {title: image_b64}

    return render_template(
        "preprocessing_result.html",
        filename=filename,
        shape=cleaned_df.shape,
        preview_data=cleaned_df.head(10).to_dict(orient="records"),
        columns=cleaned_df.columns.tolist(),
        eda_report=eda_report,   # {title: image_b64}
    )


# ── On-demand chart AI analysis ────────────────────────────────────────────────
@app.route("/analyse_chart", methods=["POST"])
def analyse_chart_route():
    """
    Receives {image_b64, chart_title} from the frontend JS.
    Calls Oxlo vision API for that one chart.
    Returns JSON {represents, key_findings, anomalies, recommendations}.
    """
    data        = request.get_json()
    image_b64   = data.get("image_b64", "")
    chart_title = data.get("chart_title", "")

    if not image_b64 or not chart_title:
        return jsonify({"error": "Missing image_b64 or chart_title"}), 400

    from src.components.eda_processing import DataPreprocessing
    from src.utils import analyse_chart, empty_analysis

    oxlo_key = os.environ.get("OXLO_API_KEY", "")
    oxlo_url = "https://api.oxlo.ai/v1/chat/completions"
    oxlo_model = "mistral-7b"

    try:
        result = analyse_chart(image_b64, chart_title, oxlo_key, oxlo_url, oxlo_model)
        return jsonify(result)
    except Exception as e:
        return jsonify(empty_analysis(chart_title)), 200
    
@app.route("/ml", methods=["GET", "POST"])
def ml_pipeline_route():
    filename = request.args.get("filename")
    if not filename:
        return redirect(url_for("index"))

    if request.method == "GET":
        from src.utils import load_dataframe_from_mongo
        df = load_dataframe_from_mongo(filename)
        return render_template(
            "ml_input.html",
            filename=filename,
            columns=df.columns.tolist(),
        )

    target_column = request.form.get("target_column", "")
    if not target_column:
        return redirect(url_for("ml_pipeline_route", filename=filename))

    from src.components.ml_pipeline import MLPipeline
    pipeline = MLPipeline(filename=filename, target_column=target_column)
    output   = pipeline.run()

    return render_template(
        "ml_results.html",
        filename=filename,
        target_column=target_column,
        problem_type=output["problem_type"],
        feature_plan=output["feature_plan"],
        results=output["results"],
        best_model=output["best_model"],
        rank_metric=output["rank_metric"],
        mongo_doc_id=output["mongo_doc_id"],
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)