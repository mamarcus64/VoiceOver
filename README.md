# VoiceOver
Annotation website for the SAIL VOICES project.

Simple, modular site for small-team annotation projects (≤3 simultaneous users,
≤6 short videos per page).

---

## 1. Quick start

```bash
# 1 Install
conda create -n annotate python=3.11 -y
conda activate annotate
pip install -r requirements.txt

# 2 Set env var so Flask can find the app
export FLASK_APP=app:app

# 3 Run (auto-reload on save)
flask run --port 5000
```

Go to **[http://localhost:5000/count\_subjects/00000](http://localhost:5000/count_subjects/00000)** to start.

---

## 2. Project layout

```
annotation_site/
│
├── app.py                # Flask entry-point + routing
├── tasks.py              # AnnotationTask base + all concrete tasks
├── templates/
│   ├── base.html         # Shared layout (header, progress bar, JS)
│   └── annotate.html     # Per-stimulus page
├── static/
│   ├── main.css
│   └── main.js
├── requirements.txt
└── results/              # Filled automatically (annotator/Task.csv)
```

---

## 3. Base class: `AnnotationTask`

```python
# tasks.py
from pathlib import Path
import pandas as pd
from moviepy.editor import VideoFileClip
import cv2

class AnnotationTask:
    """
    Subclass this once per task.
    """
    name: str                 # e.g. "count_subjects"
    stimuli: pd.DataFrame     # must include 'stimulus_id' and 'filepath'
    options: list             # list[Label] — see below

    ## ---- mandatory API --------------------------------
    def render_stimuli(self, row: pd.Series) -> list:
        """
        Return [img_or_vid_or_text, ...] to be shown on the page for that row.
        • image:  np.ndarray (H, W, 3)  BGR
        • video:  np.ndarray (T, H, W, 3)  BGR
        • text:   str
        """
        raise NotImplementedError
```

### Mini label helpers

```python
class ChooseOne:  def __init__(self, label, choices, required=False): ...
class ChooseMany: def __init__(self, label, choices, required=False): ...
class FreeText:   def __init__(self, label, placeholder='', required=False): ...
```

---

## 4. Example task: **`count_subjects`**

```python
class CountSubjects(AnnotationTask):
    name = "count_subjects"

    # 1 build stimuli table from folder contents
    folder = Path("/data2/mjma/voices/test_data/videos")
    files  = sorted(folder.glob("*.mp4"))
    stimuli = pd.DataFrame({
        "stimulus_id": [f"{i:05d}" for i in range(len(files))],
        "filepath":    [str(f) for f in files],
    })

    # 2 task-specific options
    options = [
        ChooseOne("Number of Subjects", ["0", "1", "2", "3"], required=True)
    ]

    # 3 render = 4 thumbnails @ 20/40/60/80 %
    def render_stimuli(self, row):
        fp = row["filepath"]
        clip = VideoFileClip(fp)
        duration = clip.duration
        frames = []
        for ratio in (0.2, 0.4, 0.6, 0.8):
            t = duration * ratio
            frame = clip.get_frame(t)           # RGB H×W×3
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
        clip.reader.close()
        return frames   # list[np.ndarray]
```

After defining the task, register it in **app.py**:

```python
from tasks import CountSubjects
TASKS = {t.name: t for t in [CountSubjects()]}
```

---

## 5. `app.py` (core routing)

```python
from flask import Flask, render_template, request, redirect, url_for, abort
import csv, os

app = Flask(__name__)
# after TASKS dict is defined …

@app.route("/<task>/<stim_id>")
def annotate(task, stim_id):
    t = TASKS.get(task) or abort(404)
    df = t.stimuli
    row = df.loc[df.stimulus_id == stim_id].squeeze()
    renderables = t.render_stimuli(row)        # list for template
    progress = f"{int(stim_id)+1}/{len(df)}"
    return render_template("annotate.html",
                           task=t,
                           stim_id=stim_id,
                           renderables=renderables,
                           progress=progress)

@app.post("/submit/<task>/<stim_id>")
def submit(task, stim_id):
    t = TASKS.get(task) or abort(404)
    form = request.form
    annotator = form.get("annotator", "").strip()
    if not annotator:
        return "Annotator name required", 400
    # simple required-field check
    for opt in t.options:
        if opt.required and opt.label not in form:
            return "Missing required fields", 400
    # 1 append to CSV
    out_dir = f"results/{annotator}"
    os.makedirs(out_dir, exist_ok=True)
    out_fp = f"{out_dir}/{task}.csv"
    write_hdr = not os.path.exists(out_fp)
    with open(out_fp, "a", newline="") as f:
        w = csv.writer(f)
        if write_hdr:
            w.writerow(["stimulus_id"] + [o.label for o in t.options] +
                       ["notes", "unsure"])
        w.writerow([stim_id] +
                   [form.get(o.label, "") for o in t.options] +
                   [form.get("notes", ""), form.get("unsure", "off")])
    # 2 navigation
    next_id = int(stim_id) + (1 if "next" in form else -1)
    if next_id >= len(t.stimuli):
        return redirect(url_for("thanks"))
    next_id = f"{next_id:05d}"
    return redirect(url_for("annotate", task=task, stim_id=next_id))

@app.get("/thanks")
def thanks(): return "<h2>Thank you for annotating!</h2>"
```

---

## 6. Templates

### `templates/base.html`

```html
<!doctype html><html lang="en"><head>
  <meta charset="utf-8">
  <link rel="stylesheet" href="{{ url_for('static', filename='main.css') }}">
</head><body>{% block body %}{% endblock %}
<script src="{{ url_for('static', filename='main.js') }}"></script>
</body></html>
```

### `templates/annotate.html`

```html
{% extends "base.html" %}
{% block body %}
<div class="pad"></div>

<header>
  <h2>{{ task.name }} – {{ stim_id }}</h2>
  <div class="progress">{{ progress }}</div>
</header>

<section class="stimuli">
  {% for item in renderables %}
    {% if item is string %}
        <p class="stim-text">{{ item }}</p>
    {% else %}
        <img src="data:image/jpeg;base64,{{ item|b64encode }}" class="stim-img">
    {% endif %}
  {% endfor %}
</section>

<form method="post" action="{{ url_for('submit', task=task.name, stim_id=stim_id) }}">
  <section class="options">
    {% for opt in task.options %}
      <div class="label-block">
        <p>{{ opt.label }}{% if opt.required %}*{% endif %}</p>
        {% if opt.__class__.__name__ == 'ChooseOne' %}
          {% for c in opt.choices %}
            <button type="button" class="choice" data-target="{{ opt.label }}"
                    onclick="toggleRadio(this)">{{ c }}</button>
          {% endfor %}
          <input type="hidden" name="{{ opt.label }}">
        {% endif %}
        <!-- (Analogous markup for ChooseMany / FreeText) -->
      </div>
    {% endfor %}
  </section>

  <section class="submit">
    <input name="annotator" placeholder="Annotator name" value="{{ request.form.annotator or '' }}">
    <textarea name="notes" placeholder="Notes…"></textarea>
    <label><input type="checkbox" name="unsure"> Unsure</label>
    <button name="prev">Back</button>
    <button name="next">Submit</button>
  </section>
</form>
{% endblock %}
```

*(The `|b64encode` filter can be added with a helper to embed thumbnails; full
video display is similar using `<video>` tags.)*

---

## 7. Static resources

### `static/main.css`

```css
body { font-family: sans-serif; margin:0 }
.pad { height:5vh }
header { height:10vh; display:flex; justify-content:space-between; align-items:center; padding:0 2rem }
.progress { font-size:1.2rem }
.stimuli { height:50vh; display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:1rem }
.stim-img { max-height:45vh; max-width:30% }
.options { height:20vh; display:flex; flex-wrap:wrap; gap:1rem; padding:1rem }
.choice { padding:.5rem 1rem; border:1px solid #666; border-radius:6px; background:#eee; cursor:pointer }
.choice.active { background:#5fa; }
.submit { height:15vh; display:flex; flex-direction:column; gap:.5rem; padding:1rem }
```

### `static/main.js`

```js
/* Toggle logic for ChooseOne buttons */
function toggleRadio(btn) {
  const target = btn.dataset.target;
  /* turn off siblings */
  btn.parentElement.querySelectorAll(".choice").forEach(b => {
    b.classList.remove("active");
  });
  /* activate this one and copy value to hidden input */
  btn.classList.add("active");
  btn.parentElement.querySelector(`input[name="${target}"]`).value = btn.textContent.trim();
}
```

---

## 8. Requirements

```
Flask==3.0.*
moviepy==1.0.*
opencv-python-headless==4.*
pandas==2.*
```

*`moviepy` needs **ffmpeg** in `PATH` (conda-forge or system install).*

---

## 9. Extending with new tasks

1. **Subclass** `AnnotationTask` in `tasks.py`.
2. Implement `render_stimuli` and set `options`.
3. Add the instance to `TASKS` in `app.py`.
4. Run `flask run` and visit `/<name>/00000`.

---

## 10. Known limits / future ideas

* No user authentication; blank annotator names are rejected.
* Concurrent writes are safe because each annotator has a private CSV.
* Thumbnails are generated on every page load; cache them if you notice lag.
* Progress bar is static (stimulus index / total); add a DB to track completion if needed.