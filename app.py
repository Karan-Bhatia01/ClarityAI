from flask import (
    Flask, 
    render_template, 
    request, 
    redirect, 
    url_for
)
from src.components.data_ingestion import DataIngestion
from src.components.data_info import DataInfo

app = Flask(__name__)

data_ingestion = DataIngestion()


@app.route("/", methods=["GET", "POST"])
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
        columns=columns
    )

@app.route("/info")
def info_layer():

    filename = request.args.get("filename")
    data_information = DataInfo(filename=filename)

    analysis = data_information.dataset_analysis()
    unique = data_information.get_unique_column_values()
    
    return render_template(
        "info.html",
        analysis = analysis,
        unique = unique
    )
    

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)