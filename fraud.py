
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
import hashlib
import json
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
app.secret_key = "kerala_geo_shield_2025"

# --- DATA STORAGE (In-Memory) ---
user_db = {"admin": {"password": "admin123", "name": "System Admin"}}
policy_db = []

# --- PART 1: AI ENGINE ---
def train_model():
    print("AI Engine: Training with Kerala Regional Data...")
    np.random.seed(42); n = 2000
    data = {
        'Age': np.random.randint(18, 70, n),
        'Claim_Amount': np.random.uniform(500, 25000, n),
        'Policy_Type': np.random.choice(['Auto', 'Home', 'Life'], n),
        'Days_Since_Purchase': np.random.randint(1, 1000, n),
        'Region': np.random.choice(['Trivandrum', 'Kochi', 'Kozhikode', 'Munnar'], n),
        'Fraud_Reported': np.random.choice([0, 1], n, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data); X = df.drop('Fraud_Reported', axis=1); y = df['Fraud_Reported']
    ct = ColumnTransformer([('num', StandardScaler(), [0,1,3]), ('cat', OneHotEncoder(), [2,4])])
    X_res, y_res = SMOTE().fit_resample(ct.fit_transform(X), y)
    model = xgb.XGBClassifier(eval_metric='logloss').fit(X_res, y_res)
    return model, ct

MODEL, PREPROCESSOR = train_model()

# --- PART 2: LOGIN & REGISTER UI ---
LOGIN_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Shield | Kerala Gateway</title>
    <style>
        :root { --primary: #6366f1; --bg: #0b0f1a; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: white; height: 100vh; display: flex; justify-content: center; align-items: center; margin: 0; }
        .card { background: rgba(30, 41, 59, 0.8); padding: 30px; border-radius: 20px; text-align: center; width: 380px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px); }
        input { width: 100%; padding: 12px; margin: 8px 0; background: #161b22; border: 1px solid #334155; border-radius: 8px; color: white; box-sizing: border-box; }
        button { width: 100%; padding: 12px; background: var(--primary); border: none; border-radius: 8px; color: white; font-weight: bold; cursor: pointer; margin-top: 10px; }
        .toggle { color: #818cf8; cursor: pointer; font-size: 13px; margin-top: 15px; display: block; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="card">
        <h2 id="title">🛡️ LOGIN</h2>
        <form id="loginForm" action="/auth" method="POST">
            <input type="text" name="u" placeholder="Username" required>
            <input type="password" name="p" placeholder="Password" required>
            <button type="submit">Enter Shield</button>
            <span class="toggle" onclick="toggle()">New User? Register here</span>
        </form>
        <form id="regForm" class="hidden" action="/register_user" method="POST">
            <input type="text" name="u" placeholder="New Username" required>
            <input type="password" name="p" placeholder="New Password" required>
            <input type="text" name="name" placeholder="Full Name" required>
            <button type="submit" style="background:#10b981">Create Account</button>
            <span class="toggle" onclick="toggle()">Back to Login</span>
        </form>
    </div>
    <script>
        function toggle() {
            document.getElementById('loginForm').classList.toggle('hidden');
            document.getElementById('regForm').classList.toggle('hidden');
            document.getElementById('title').innerText = document.getElementById('title').innerText.includes("LOGIN") ? "📝 REGISTER" : "🛡️ LOGIN";
        }
    </script>
</body>
</html>
"""

# --- PART 3: MAIN DASHBOARD (MAP + POLICY REG) ---
DASHBOARD_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>Shield Pro | Kerala Analytics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root { --primary: #6366f1; --bg: #0b0f1a; --glass: rgba(255, 255, 255, 0.05); }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: #e2e8f0; margin: 0; padding: 20px; }
        .container { max-width: 1300px; margin: auto; }
        .dashboard { display: grid; grid-template-columns: 400px 1fr; gap: 20px; }
        .card { background: var(--glass); border: 1px solid rgba(255,255,255,0.1); padding: 25px; border-radius: 20px; backdrop-filter: blur(10px); }
        .tabs { grid-column: 1 / span 2; display: flex; gap: 15px; border-bottom: 1px solid #1e293b; padding-bottom: 15px; margin-bottom: 20px; }
        .tab-btn { cursor: pointer; padding: 10px 20px; border-radius: 8px; color: #94a3b8; transition: 0.3s; }
        .tab-btn.active { background: var(--primary); color: white; }
        .section { display: none; } .section.active { display: block; }
        input, select { width: 100%; padding: 12px; margin: 8px 0; background: #161b22; border: 1px solid #30363d; border-radius: 8px; color: white; box-sizing: border-box; }
        #map { height: 450px; border-radius: 15px; border: 1px solid #30363d; margin-top: 15px; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 11px; margin: 3px; background: rgba(99, 102, 241, 0.2); border: 1px solid var(--primary); }
    </style>
</head>
<body>
    <div class="container">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
            <h2>🛡️ Shield Pro <span style="color:var(--primary)">KERALA</span></h2>
            <div style="font-size:14px; color:#94a3b8;">User: {{ user_name }} | <a href="/logout" style="color:#f87171; text-decoration:none;">Logout</a></div>
        </div>

        <div class="tabs">
            <div id="btn-audit" class="tab-btn active" onclick="showTab('audit')"><i class="fas fa-microchip"></i> Fraud Audit</div>
            <div id="btn-policy" class="tab-btn" onclick="showTab('policy')"><i class="fas fa-file-contract"></i> Policy Registration</div>
        </div>

        <div class="dashboard">
            <div class="card">
                <div id="auditSection" class="section active">
                    <h3>Audit Claim</h3>
                    <label>Claim Amount (₹)</label><input type="number" id="Amount" value="25000">
                    <label>Days Held</label><input type="number" id="Days" value="15">
                    <label>Location (Kerala)</label>
                    <select id="Region">
                        <option value="Trivandrum">Trivandrum</option>
                        <option value="Kochi">Kochi</option>
                        <option value="Kozhikode">Kozhikode</option>
                        <option value="Munnar">Munnar</option>
                    </select>
                    <button onclick="runKeralaAudit()" style="width:100%; padding:12px; background:var(--primary); border:none; border-radius:8px; color:white; font-weight:bold; cursor:pointer;">Analyze Risk</button>
                    <div style="margin-top:25px; text-align:center;">
                        <canvas id="gaugeChart" style="max-height:120px;"></canvas>
                        <h2 id="scoreDisplay">--%</h2>
                    </div>
                </div>

                <div id="policySection" class="section">
                    <h3>New Policy</h3>
                    <input type="text" id="pName" placeholder="Policy Holder Name">
                    <input type="number" id="pAge" placeholder="Age">
                    <select id="pType"><option>Auto</option><option>Home</option><option>Life</option></select>
                    <button onclick="savePolicy()" style="width:100%; padding:12px; background:#10b981; border:none; border-radius:8px; color:white; font-weight:bold; cursor:pointer;">Register Policy</button>
                </div>
            </div>

            <div class="card">
                <h3>Kerala Fraud Hotspots</h3>
                <div id="map"></div>
                <div id="explanations" style="margin-top:15px;"></div>
                <small id="tokenDisplay" style="font-family:monospace; font-size:9px; color:#64748b; word-break:break-all;"></small>
            </div>
        </div>
    </div>

    <script>
        let gauge, map, marker;
        const keralaCoords = {
            'Trivandrum': [8.5241, 76.9366],
            'Kochi': [9.9312, 76.2673],
            'Kozhikode': [11.2588, 75.7804],
            'Munnar': [10.0889, 77.0595]
        };

        window.onload = () => {
            const ctx = document.getElementById('gaugeChart').getContext('2d');
            gauge = new Chart(ctx, { type: 'doughnut', data: { datasets: [{ data: [0, 100], backgroundColor: ['#6366f1', '#1e293b'], borderWidth: 0, circumference: 180, rotation: 270, cutout: '85%' }] }, options: { animation: { duration: 1000 }, plugins: { tooltip: { enabled: false } } } });

            map = L.map('map').setView([10.8505, 76.2711], 7); // Center of Kerala
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);
        };

        function showTab(t) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(t + 'Section').classList.add('active');
            document.getElementById('btn-' + t).classList.add('active');
        }

        function savePolicy() {
            alert("Policy registered successfully for " + document.getElementById('pName').value);
            showTab('audit');
        }

        async function runKeralaAudit() {
            const region = document.getElementById('Region').value;
            const payload = { Age: 30, Claim_Amount: parseFloat(document.getElementById('Amount').value), Policy_Type: 'Auto', Days_Since_Purchase: parseInt(document.getElementById('Days').value), Region: region };

            const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await res.json();

            gauge.data.datasets[0].data = [data.score, 100 - data.score];
            gauge.data.datasets[0].backgroundColor = [data.score > 70 ? '#ef4444' : '#22c55e', '#1e293b'];
            gauge.update();
            document.getElementById('scoreDisplay').innerText = data.score + "%";
            document.getElementById('tokenDisplay').innerText = "Audit Hash: " + data.token;
            document.getElementById('explanations').innerHTML = data.reasons.map(r => `<span class="badge">${r}</span>`).join('');

            if (marker) map.removeLayer(marker);
            const coords = keralaCoords[region];
            map.flyTo(coords, 10);
            marker = L.circleMarker(coords, { color: data.score > 70 ? '#ef4444' : '#22c55e', radius: 15, fillOpacity: 0.6 }).addTo(map).bindPopup(`<b>Risk: ${data.score}%</b><br>Location: ${region}`).openPopup();
        }
    </script>
</body>
</html>
"""

# --- PART 4: FLASK LOGIC ---
@app.route('/')
def index(): return render_template_string(LOGIN_UI)

@app.route('/auth', methods=['POST'])
def auth():
    u, p = request.form.get('u'), request.form.get('p')
    if u in user_db and user_db[u]['password'] == p:
        session['logged_in'], session['user_name'] = True, user_db[u]['name']
        return redirect(url_for('dashboard'))
    return redirect('/')

@app.route('/register_user', methods=['POST'])
def register_user():
    u, p, name = request.form.get('u'), request.form.get('p'), request.form.get('name')
    user_db[u] = {"password": p, "name": name}
    return redirect('/')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'): return redirect('/')
    return render_template_string(DASHBOARD_UI, user_name=session.get('user_name'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    processed = PREPROCESSOR.transform(pd.DataFrame([data]))
    score = round(float(MODEL.predict_proba(processed)[:, 1][0]) * 100, 1)
    reasons = ["⚠️ High Value" if data['Claim_Amount'] > 15000 else "✅ Normal Value"]
    if data['Days_Since_Purchase'] < 30: reasons.append("🕒 Rapid Claim")
    token = hashlib.sha256(f"{data}-{score}".encode()).hexdigest()
    return jsonify({'score': score, 'reasons': reasons, 'token': token})

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)