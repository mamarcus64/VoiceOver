"""
Flask application for annotation website
"""
from flask import Flask, render_template, request, redirect, url_for, abort, jsonify
import csv
import os
import base64
import cv2
import io
from datetime import datetime
import pandas as pd
from tasks import CountSubjects

app = Flask(__name__)

# Register all tasks
TASKS = {t.name: t for t in [CountSubjects()]}


# Custom template filter to convert numpy arrays to base64
@app.template_filter('b64encode')
def b64encode_filter(data):
    """Convert numpy array (image) to base64 string for embedding in HTML"""
    if isinstance(data, str):
        return data
    
    # Encode image as JPEG
    success, buffer = cv2.imencode('.jpg', data)
    if success:
        # Convert to base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text
    return ""


@app.route("/")
def index():
    """Homepage - redirect to first task"""
    if TASKS:
        first_task = list(TASKS.keys())[0]
        return redirect(url_for('annotate', task=first_task, stim_id='00000'))
    return "No tasks configured", 404


@app.route("/<task>")
def task_start(task):
    """Redirect to first stimulus of a task"""
    if task in TASKS:
        return redirect(url_for('annotate', task=task, stim_id='00000'))
    abort(404)


@app.route("/<task>/<stim_id>")
def annotate(task, stim_id):
    """Main annotation page for a specific stimulus"""
    t = TASKS.get(task)
    if not t:
        abort(404)
    
    # Get stimulus data
    df = t.stimuli
    matching_rows = df.loc[df.stimulus_id == stim_id]
    
    if matching_rows.empty:
        abort(404)
    
    row = matching_rows.iloc[0]
    
    # Render stimuli (images/videos/text)
    try:
        renderables = t.render_stimuli(row)
    except Exception as e:
        print(f"Error rendering stimuli: {e}")
        renderables = []
    
    # Calculate progress
    current_idx = int(stim_id) + 1
    total = len(df)
    progress = f"{current_idx}/{total}"
    
    return render_template("annotate.html",
                         task=t,
                         stim_id=stim_id,
                         renderables=renderables,
                         progress=progress,
                         annotator=request.cookies.get('annotator', ''))


@app.post("/submit/<task>/<stim_id>")
def submit(task, stim_id):
    """Handle form submission"""
    t = TASKS.get(task)
    if not t:
        abort(404)
    
    form = request.form
    annotator = form.get("annotator", "").strip()
    
    # Validate annotator name
    if not annotator:
        return "Annotator name required", 400
    
    # Validate required fields
    for opt in t.options:
        if opt.required:
            value = form.get(opt.label, "").strip()
            if not value:
                return f"Missing required field: {opt.label}", 400
    
    # Create output directory
    out_dir = f"results/{annotator}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Prepare CSV file
    out_fp = f"{out_dir}/{task}.csv"
    write_header = not os.path.exists(out_fp)
    
    # Write results
    with open(out_fp, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header if needed
        if write_header:
            headers = ["stimulus_id"] + [opt.label for opt in t.options] + ["notes", "unsure", "timestamp"]
            writer.writerow(headers)
        
        # Get current timestamp in readable format
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write data row
        row_data = [stim_id]
        for opt in t.options:
            row_data.append(form.get(opt.label, ""))
        row_data.append(form.get("notes", ""))
        row_data.append("yes" if form.get("unsure") == "on" else "no")
        row_data.append(timestamp)
        
        writer.writerow(row_data)
    
    # Determine next stimulus
    if "prev" in form:
        # Go back
        next_idx = int(stim_id) - 1
        if next_idx < 0:
            next_idx = 0
    else:
        # Go forward
        next_idx = int(stim_id) + 1
        if next_idx >= len(t.stimuli):
            return redirect(url_for("thanks", annotator=annotator))
    
    next_id = f"{next_idx:05d}"
    
    # Set cookie for annotator name and redirect
    response = redirect(url_for("annotate", task=task, stim_id=next_id))
    response.set_cookie('annotator', annotator)
    return response


@app.route("/next_unfilled/<task>/<start_id>")
def next_unfilled(task, start_id):
    """Find the next unfilled task starting from a given index"""
    t = TASKS.get(task)
    if not t:
        abort(404)
    
    annotator = request.args.get('annotator', '')
    if not annotator:
        return jsonify({"found": False, "error": "No annotator specified"})
    
    scope = request.args.get('scope', 'you')  # 'you' or 'any'
    
    filled_ids = set()
    
    if scope == 'you':
        # Check only the current annotator's results
        csv_path = f"results/{annotator}/{task}.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'stimulus_id' in df.columns:
                    filled_ids = set(df['stimulus_id'].astype(str))
            except Exception as e:
                print(f"Error reading CSV: {e}")
    else:
        # Check all annotators' results
        results_dir = "results"
        if os.path.exists(results_dir):
            for annotator_dir in os.listdir(results_dir):
                csv_path = os.path.join(results_dir, annotator_dir, f"{task}.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if 'stimulus_id' in df.columns:
                            filled_ids.update(df['stimulus_id'].astype(str))
                    except Exception as e:
                        print(f"Error reading CSV {csv_path}: {e}")
    
    # Get all stimulus IDs from the task and ensure they're strings
    all_ids = [str(sid) for sid in t.stimuli['stimulus_id']]
    
    # Ensure start_id is a string and properly formatted
    start_id = str(start_id).zfill(5)
    
    # Find the starting index
    try:
        start_idx = all_ids.index(start_id)
    except ValueError:
        # If start_id not found, try to convert it to int and back to padded string
        try:
            start_num = int(start_id)
            start_id_padded = f"{start_num:05d}"
            start_idx = all_ids.index(start_id_padded)
        except (ValueError, IndexError):
            start_idx = 0
    
    # Search for the next unfilled task starting from start_idx
    for i in range(start_idx, len(all_ids)):
        if all_ids[i] not in filled_ids:
            return jsonify({"found": True, "stimulus_id": all_ids[i]})
    
    # If nothing found from start_idx to end, search from beginning up to start_idx
    for i in range(0, start_idx):
        if all_ids[i] not in filled_ids:
            return jsonify({"found": True, "stimulus_id": all_ids[i]})
    
    return jsonify({"found": False})


@app.route("/thanks")
def thanks():
    """Thank you page after completing all annotations"""
    annotator = request.args.get('annotator', 'Anonymous')
    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f0f0;
            }}
            .message {{
                text-align: center;
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="message">
            <h1>Thank you, {annotator}!</h1>
            <p>You have completed all annotations.</p>
            <a href="/">Return to start</a>
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(debug=True, port=5000)