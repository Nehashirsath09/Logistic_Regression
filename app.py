"""
GenAI Learning-Impact Predictor  -  Flask app for Render
Model: scikit-learn LogisticRegression (classes: High / Low / Medium)

IMPORTANT: the model was trained on 11 *numeric* columns. For the categorical
columns the app must send the same integer codes that were used in training.
Edit the CATEGORIES lists below so they match your training data.
Codes are assigned by position in the SORTED list (same as sklearn's LabelEncoder).
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, os.environ.get("MODEL_FILE", "e6f2842c-logistic.pkl"))
model = joblib.load(MODEL_PATH)

# ----------------------------------------------------------------------------
# Feature definitions (order is taken from the model itself, so it can't drift)
# ----------------------------------------------------------------------------
CATEGORIES = {  # <-- EDIT these to match the labels in your training data
    "Major_Category": ["Arts", "Business", "Engineering", "Humanities", "Medical", "Science"],
    "Primary_Use_Case": ["Brainstorming", "Coding", "Research", "Summarizing", "Writing"],
    "Paid_Subscription": ["No", "Yes"],
    "Institutional_Policy": ["Allowed", "Encouraged", "Restricted"],
}
for _k in CATEGORIES:
    CATEGORIES[_k] = sorted(CATEGORIES[_k])

NUMERIC = {  # name: (label, min, max, step, default)
    "Year_of_Study": ("Year of study", 1, 5, 1, 2),
    "Pre_Semester_GPA": ("Pre-semester GPA", 0, 4, 0.01, 3.0),
    "Weekly_GenAI_Hours": ("Weekly GenAI hours", 0, 60, 0.5, 10),
    "Prompt_Engineering_Skill": ("Prompt engineering skill (1-10)", 1, 10, 1, 5),
    "Traditional_Study_Hours": ("Traditional study hours / week", 0, 60, 0.5, 15),
    "Perceived_AI_Dependency": ("Perceived AI dependency (1-10)", 1, 10, 1, 5),
    "Skill_Retention_Score": ("Skill retention score (0-100)", 0, 100, 1, 60),
}
LABELS = {
    "Major_Category": "Major",
    "Primary_Use_Case": "Primary use case",
    "Paid_Subscription": "Paid subscription",
    "Institutional_Policy": "Institutional AI policy",
}

FEATURES = list(model.feature_names_in_)
CLASSES = [str(c) for c in model.classes_]


def build_schema():
    schema = []
    for f in FEATURES:
        if f in CATEGORIES:
            schema.append({"name": f, "label": LABELS.get(f, f), "type": "select",
                           "options": CATEGORIES[f]})
        else:
            label, lo, hi, step, default = NUMERIC.get(f, (f, 0, 100, 1, 0))
            schema.append({"name": f, "label": label, "type": "range",
                           "min": lo, "max": hi, "step": step, "default": default})
    return schema


SCHEMA = build_schema()

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", features=FEATURES, classes=CLASSES)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    row = {}
    try:
        for f in FEATURES:
            if f not in data:
                return jsonify(error=f"Missing field: {f}"), 400
            if f in CATEGORIES:
                row[f] = CATEGORIES[f].index(str(data[f]))
            else:
                row[f] = float(data[f])
    except (ValueError, TypeError):
        return jsonify(error="Invalid value in the form."), 400

    X = pd.DataFrame([row], columns=FEATURES)
    proba = model.predict_proba(X)[0]
    best = int(np.argmax(proba))
    return jsonify(
        prediction=CLASSES[best],
        confidence=round(float(proba[best]) * 100, 1),
        probabilities={c: round(float(p) * 100, 1) for c, p in zip(CLASSES, proba)},
    )


@app.get("/")
def index():
    return render_template_string(PAGE, schema=SCHEMA)


PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GenAI Learning Impact Predictor</title>
<style>
  :root{--bg:#0b1020;--card:#141a33;--line:#273056;--text:#e8ebff;--muted:#8d97c4;
        --a:#7c5cff;--b:#22d3ee;--high:#34d399;--med:#fbbf24;--low:#f87171}
  *{box-sizing:border-box}
  body{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--text);
       background:radial-gradient(900px 500px at 10% -10%,#2a1f6b55,transparent),
                  radial-gradient(800px 500px at 100% 0%,#0e7490464,transparent),var(--bg);min-height:100vh}
  .wrap{max-width:1040px;margin:0 auto;padding:40px 20px 60px}
  h1{font-size:clamp(1.7rem,4vw,2.4rem);margin:0 0 6px;
     background:linear-gradient(90deg,var(--a),var(--b));-webkit-background-clip:text;color:transparent}
  .sub{color:var(--muted);margin:0 0 28px}
  .grid{display:grid;grid-template-columns:1.4fr 1fr;gap:22px}
  @media(max-width:860px){.grid{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:24px;
        box-shadow:0 10px 40px #0006}
  .fields{display:grid;grid-template-columns:1fr 1fr;gap:18px 20px}
  @media(max-width:560px){.fields{grid-template-columns:1fr}}
  label{display:flex;justify-content:space-between;font-size:.85rem;color:var(--muted);margin-bottom:7px}
  label b{color:var(--text)}
  select{width:100%;padding:11px 12px;border-radius:10px;border:1px solid var(--line);
         background:#0e1430;color:var(--text);font-size:.95rem}
  input[type=range]{width:100%;accent-color:var(--a)}
  button{width:100%;margin-top:24px;padding:14px;border:0;border-radius:12px;font-size:1rem;font-weight:600;
         color:#fff;cursor:pointer;background:linear-gradient(90deg,var(--a),var(--b));transition:.2s}
  button:hover{transform:translateY(-1px);filter:brightness(1.1)}
  button:disabled{opacity:.6;cursor:wait}
  .result{text-align:center}
  .badge{display:inline-block;font-size:2.4rem;font-weight:800;margin:10px 0 2px}
  .High{color:var(--high)}.Medium{color:var(--med)}.Low{color:var(--low)}
  .conf{color:var(--muted);margin-bottom:22px}
  .bar{text-align:left;margin:12px 0}
  .bar .top{display:flex;justify-content:space-between;font-size:.85rem;margin-bottom:5px}
  .track{height:10px;background:#0e1430;border-radius:99px;overflow:hidden}
  .fill{height:100%;width:0;border-radius:99px;transition:width .8s cubic-bezier(.2,.8,.2,1)}
  .fill.High{background:var(--high)}.fill.Medium{background:var(--med)}.fill.Low{background:var(--low)}
  .placeholder{color:var(--muted);padding:50px 10px}
  .err{color:var(--low)}
  footer{margin-top:26px;text-align:center;color:var(--muted);font-size:.8rem}
</style>
</head>
<body>
<div class="wrap">
  <h1>GenAI Learning Impact Predictor</h1>
  <p class="sub">Enter a student's profile and the logistic-regression model predicts the outcome class.</p>
  <div class="grid">
    <form id="form" class="card">
      <div class="fields">
      {% for f in schema %}
        <div>
          {% if f.type == 'select' %}
            <label>{{ f.label }}</label>
            <select name="{{ f.name }}">
              {% for o in f.options %}<option>{{ o }}</option>{% endfor %}
            </select>
          {% else %}
            <label>{{ f.label }} <b id="v_{{ f.name }}">{{ f.default }}</b></label>
            <input type="range" name="{{ f.name }}" min="{{ f.min }}" max="{{ f.max }}"
                   step="{{ f.step }}" value="{{ f.default }}"
                   oninput="document.getElementById('v_{{ f.name }}').textContent=this.value">
          {% endif %}
        </div>
      {% endfor %}
      </div>
      <button id="btn" type="submit">Predict</button>
    </form>

    <div class="card result" id="out">
      <div class="placeholder">Your prediction will appear here.</div>
    </div>
  </div>
  <footer>Logistic Regression &middot; scikit-learn &middot; deployed on Render</footer>
</div>

<script>
const form = document.getElementById('form'), out = document.getElementById('out'), btn = document.getElementById('btn');
form.addEventListener('submit', async e => {
  e.preventDefault(); btn.disabled = true; btn.textContent = 'Predicting...';
  const body = Object.fromEntries(new FormData(form).entries());
  try {
    const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Request failed');
    const order = ['High','Medium','Low'].filter(k => k in d.probabilities)
                  .concat(Object.keys(d.probabilities).filter(k => !['High','Medium','Low'].includes(k)));
    out.innerHTML = `<div class="badge ${d.prediction}">${d.prediction}</div>
      <div class="conf">${d.confidence}% confidence</div>` +
      order.map(k => `<div class="bar"><div class="top"><span>${k}</span><span>${d.probabilities[k]}%</span></div>
        <div class="track"><div class="fill ${k}" data-w="${d.probabilities[k]}"></div></div></div>`).join('');
    requestAnimationFrame(() => out.querySelectorAll('.fill').forEach(el => el.style.width = el.dataset.w + '%'));
  } catch (err) {
    out.innerHTML = `<div class="placeholder err">${err.message}</div>`;
  } finally { btn.disabled = false; btn.textContent = 'Predict'; }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
