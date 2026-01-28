from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os
import json
import time
from datetime import datetime
import pandas as pd


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
HISTORY_FILE = 'history.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")


@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    horse_count = 0
    processing_time = 0

    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_path = os.path.join(RESULT_FOLDER, filename)

        file.save(input_path)

        start_time = time.time()

        # class 17 — horse (COCO)
        results = model(input_path, classes=[17])
        results[0].save(output_path)

        processing_time = round(time.time() - start_time, 2)
        horse_count = len(results[0].boxes)

        result_image = output_path

        save_history(filename, horse_count, processing_time)

    return render_template(
        'index.html',
        result_image=result_image,
        horse_count=horse_count,
        processing_time=processing_time
    )


def save_history(filename, count, time_spent):
    record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "horses_detected": count,
        "processing_time": time_spent
    }

    data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except:
                data = []

    data.append(record)

    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


@app.route('/stats')
def stats():
    data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

    return render_template('stats.html', data=data)


@app.route('/report')
def report():
    if not os.path.exists(HISTORY_FILE):
        return "История пуста"

    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    report_path = "horse_report.xlsx"
    df.to_excel(report_path, index=False)

    return send_file(
        report_path,
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True)
