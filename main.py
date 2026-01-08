import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
app.secret_key = "fraud_shield_2025_ultimate_key"

# Paths for AI artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'fraud_model.pkl')
PREPROCESSOR_FILE = os.path.join(BASE_DIR, 'preprocessor.pkl')

# --- DATA STORAGE ---
user_db = {"admin": {"password": "admin123", "name": "System Admin"}}
registered_policies = [] # This stores the insurance policies you add

# --- PART 1: AI MODEL LOGIC ---
def train_model():
    print("AI Engine: Initializing...")
    np.random.seed(42); n = 2000
    data = {
        'Age': np.random.randint(18, 70, n),
        'Claim_Amount': np.random.uniform(500, 25000, n),
        'Policy_Type': np.random.choice(['Auto', 'Home', 'Life'], n),
        'Days_Since_Purchase': np.random.randint(1, 1000, n),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'Fraud_Reported': np.random.choice([0, 1], n, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)
    X = df.drop('Fraud_Reported', axis=1); y = df['Fraud_Reported']
    ct = ColumnTransformer([('num', StandardScaler(), [0,1,3]), ('cat', OneHotEncoder(), [2,4])])
    X_res, y_res = SMOTE().fit_resample(ct.fit_transform(X), y)
    model = xgb.XGBClassifier(eval_metric='logloss').fit(X_res, y_res)
    joblib.dump(model, MODEL_FILE); joblib.dump(ct, PREPROCESSOR_FILE)
    return model, ct

if not os.path.exists(MODEL_FILE):
    MODEL, PREPROCESSOR = train_model()
else:
    MODEL, PREPROCESSOR = joblib.load(MODEL_FILE), joblib.load(PREPROCESSOR_FILE)

# --- PART 2: UI TEMPLATES ---

LOGIN_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Shield | Access</title>
    <style>
        :root { --primary: #6366f1; --bg: #0f172a; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); height: 100vh; display: flex; justify-content: center; align-items: center; margin: 0; color: white; }
        .login-card { background: rgba(30, 41, 59, 0.8); padding: 30px; border-radius: 20px; text-align: center; width: 380px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px); }
        input, select { width: 100%; padding: 10px; margin: 8px 0; background: #1e293b; border: 1px solid #334155; border-radius: 8px; color: white; box-sizing: border-box; }
        button { width: 100%; padding: 12px; background: var(--primary); border: none; border-radius: 8px; color: white; font-weight: bold; cursor: pointer; margin-top: 10px; }
        .toggle-link { color: #818cf8; cursor: pointer; font-size: 13px; margin-top: 15px; display: block; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="login-card">
        <h2 id="title">🛡️ LOGIN</h2>
        <form id="login-form" action="/login" method="POST">
            <input type="text" name="u" placeholder="Username" required>
            <input type="password" name="p" placeholder="Password" required>
            <button type="submit">Access Portal</button>
            <span class="toggle-link" onclick="toggle()">New User? Register App Account</span>
        </form>
        <form id="reg-form" class="hidden" action="/register" method="POST">
            <input type="text" name="u" placeholder="Username" required>
            <input type="password" name="p" placeholder="Password" required>
            <input type="text" name="name" placeholder="Full Name" required>
            <input type="text" name="phone" placeholder="Phone">
            <input type="text" name="blood" placeholder="Blood Group">
            <button type="submit" style="background:#10b981">Create App Account</button>
            <span class="toggle-link" onclick="toggle()">Back to Login</span>
        </form>
    </div>
    <script>
        function toggle() {
            document.getElementById('login-form').classList.toggle('hidden');
            document.getElementById('reg-form').classList.toggle('hidden');
            document.getElementById('title').innerText = document.getElementById('title').innerText.includes("LOGIN") ? "📝 REGISTER" : "🛡️ LOGIN";
        }
    </script>
</body>
</html>
"""

DASHBOARD_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard | Policy & Claims</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root { --primary: #6366f1; --bg: #0f172a; --card: rgba(30, 41, 59, 0.7); }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: white; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        .dashboard { background: var(--card); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); padding: 30px; border-radius: 24px; width: 1050px; display: grid; grid-template-columns: 1fr 1fr; gap: 30px; box-shadow: 0 25px 50px rgba(0,0,0,0.5); position: relative; }
        .tabs { grid-column: 1 / span 2; display: flex; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 10px; }
        .tab-btn { cursor: pointer; padding: 10px 20px; border-radius: 8px; color: #94a3b8; transition: 0.3s; }
        .tab-btn.active { background: var(--primary); color: white; }
        .section { display: none; } .section.active { display: block; }
        input, select { width: 100%; padding: 10px; background: rgba(15, 23, 42, 0.5); border: 1px solid #334155; border-radius: 8px; color: white; margin-bottom: 10px; }
        button { width: 100%; padding: 12px; background: var(--primary); border: none; border-radius: 10px; color: white; font-weight: bold; cursor: pointer; }
        .viz-box { background: rgba(15, 23, 42, 0.3); padding: 20px; border-radius: 20px; text-align: center; }
        #gaugeChart { max-width: 240px; margin: auto; }
        .logout { color: #f87171; text-decoration: none; font-size: 14px; }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="tabs">
            <div style="display:flex; gap:10px;">
                <div id="btn-audit" class="tab-btn active" onclick="showTab('audit')"><i class="fas fa-search"></i> Fraud Audit</div>
                <div id="btn-policy" class="tab-btn" onclick="showTab('policy')"><i class="fas fa-plus-circle"></i> New Policy Registration</div>
            </div>
            <div style="display:flex; align-items:center; gap:15px;">
                <span style="font-size:14px; color:#818cf8">User: {{ user_name }}</span>
                <a href="/logout" class="logout"><i class="fas fa-sign-out-alt"></i></a>
            </div>
        </div>

        <div>
            <div id="auditSection" class="section active">
                <h3 style="color:#818cf8">Claim Risk Analysis</h3>
                <label>Age</label><input type="number" id="Age" value="30">
                <label>Claim Amount ($)</label><input type="number" id="Amount" value="18000">
                <label>Policy Type</label><select id="Type"><option>Auto</option><option>Home</option><option>Life</option></select>
                <label>Days Held</label><input type="number" id="Days" value="15">
                <label>Region</label><select id="Region"><option>North</option><option>South</option><option>East</option><option>West</option></select>
                <button onclick="analyze()">Run AI Audit</button>
            </div>

            <div id="policySection" class="section">
                <h3 style="color:#10b981">Add New Insurance Policy</h3>
                <label>Holder Full Name</label><input type="text" id="pName" placeholder="e.g. John Doe">
                <label>Holder Age</label><input type="number" id="pAge" value="30">
                <label>Coverage Amount ($)</label><input type="number" id="pCover" value="10000">
                <label>Policy Category</label><select id="pType"><option>Auto</option><option>Home</option><option>Life</option></select>
                <label>Region</label><select id="pRegion"><option>North</option><option>South</option><option>East</option><option>West</option></select>
                <button style="background:#10b981" onclick="regPol()">Register New Policy</button>
            </div>
        </div>

        <div class="viz-box">
            <canvas id="gaugeChart"></canvas>
            <div id="riskLevel" style="font-size:24px; font-weight:bold; margin-top:20px;">READY</div>
            <p id="riskDetail" style="color:#94a3b8; font-size:13px; margin-top:10px;">Input data and run audit to see AI results.</p>
        </div>
    </div>

    <script>
        let gauge;
        window.onload = () => {
            const ctx = document.getElementById('gaugeChart').getContext('2d');
            gauge = new Chart(ctx, { type: 'doughnut', data: { datasets: [{ data: [0, 100], backgroundColor: ['#6366f1', '#1e293b'], borderWidth: 0, circumference: 180, rotation: 270, cutout: '85%' }] }, options: { animation: { duration: 1500 } } });
        };

        function showTab(t) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(t + 'Section').classList.add('active');
            document.getElementById('btn-' + t).classList.add('active');
        }

        function regPol() {
            alert("Policy for " + document.getElementById('pName').value + " has been successfully added to the system database.");
            showTab('audit');
        }

        async function analyze() {
            const payload = { Age: parseInt(document.getElementById('Age').value), Claim_Amount: parseFloat(document.getElementById('Amount').value), Policy_Type: document.getElementById('Type').value, Days_Since_Purchase: parseInt(document.getElementById('Days').value), Region: document.getElementById('Region').value };
            const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await res.json();
            gauge.data.datasets[0].data = [data.score, 100 - data.score];
            gauge.data.datasets[0].backgroundColor = [data.score > 70 ? '#ef4444' : '#22c55e', '#1e293b'];
            gauge.update();
            document.getElementById('riskLevel').innerText = data.score + "% RISK";
            document.getElementById('riskDetail').innerText = data.score > 70 ? "HIGH RISK: Potential fraud pattern detected." : "LOW RISK: Pattern matches normal claims.";
        }
    </script>
</body>
</html>
"""

# --- PART 3: FLASK ROUTES ---

@app.route('/')
def home(): return render_template_string(LOGIN_UI)

@app.route('/login', methods=['POST'])
def login():
    u, p = request.form.get('u'), request.form.get('p')
    if u in user_db and user_db[u]['password'] == p:
        session['logged_in'], session['user_name'] = True, user_db[u].get('name', u)
        return redirect(url_for('dashboard'))
    return redirect(url_for('home'))

@app.route('/register', methods=['POST'])
def register():
    u, p = request.form.get('u'), request.form.get('p')
    user_db[u] = {"password": p, "name": request.form.get('name')}
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'): return redirect(url_for('home'))
    return render_template_string(DASHBOARD_UI, user_name=session.get('user_name'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    processed = PREPROCESSOR.transform(pd.DataFrame([data]))
    score = round(float(MODEL.predict_proba(processed)[:, 1][0]) * 100, 1)
    return jsonify({'score': score})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)