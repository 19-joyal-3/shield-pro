import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
import hashlib
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
app.secret_key = "global_standard_security_v3"

# --- SYSTEM CONFIGURATION ---
user_db = {"admin": {"password": "admin123", "name": "Lead Auditor"}}
audit_log = [] # Simulated Ledger

# --- PART 1: ENTERPRISE AI ENGINE ---
def train_model():
    print("[SYSTEM] Training Enterprise AI Model...")
    np.random.seed(42); n = 3000
    data = {
        'Age': np.random.randint(18, 70, n),
        'Claim_Amount': np.random.uniform(1000, 50000, n),
        'Policy_Type': np.random.choice(['Auto', 'Property', 'Health', 'Marine'], n),
        'Days_Since_Purchase': np.random.randint(1, 1000, n),
        'Region': np.random.choice(['Trivandrum', 'Kochi', 'Kozhikode', 'Munnar'], n),
        'Fraud_Reported': np.random.choice([0, 1], n, p=[0.93, 0.07])
    }
    df = pd.DataFrame(data)
    ct = ColumnTransformer([('num', StandardScaler(), [0,1,3]), ('cat', OneHotEncoder(), [2,4])])
    X_res, y_res = SMOTE().fit_resample(ct.fit_transform(df.drop('Fraud_Reported', axis=1)), df['Fraud_Reported'])
    model = xgb.XGBClassifier(eval_metric='logloss', n_estimators=150).fit(X_res, y_res)
    return model, ct

MODEL, PREPROCESSOR = train_model()

# --- PART 2: PROFESSIONAL INTERFACES ---

LOGIN_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fraud Shield Pro | Secure Access</title>
    <style>
        :root { --brand: #6366f1; --dark: #0f172a; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--dark); color: white; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .auth-container { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(15px); padding: 40px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); width: 350px; text-align: center; }
        input { width: 100%; padding: 12px; margin: 10px 0; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: white; box-sizing: border-box; }
        .btn-main { width: 100%; padding: 12px; background: var(--brand); border: none; border-radius: 8px; color: white; font-weight: 600; cursor: pointer; transition: 0.3s; }
        .btn-main:hover { filter: brightness(1.2); }
        .toggle-txt { font-size: 13px; color: #94a3b8; cursor: pointer; margin-top: 20px; display: block; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="auth-container">
        <h2 id="form-title">🛡️ Fraud Shield <span style="color:var(--brand)">Pro</span></h2>
        <form id="login-form" action="/auth" method="POST">
            <input type="text" name="u" placeholder="Username (admin)" required>
            <input type="password" name="p" placeholder="Password (admin123)" required>
            <button type="submit" class="btn-main">Authenticate Session</button>
            <span class="toggle-txt" onclick="toggle()">New Investigator? Create ID</span>
        </form>
        <form id="reg-form" class="hidden" action="/register" method="POST">
            <input type="text" name="u" placeholder="Set Username" required>
            <input type="password" name="p" placeholder="Set Password" required>
            <input type="text" name="name" placeholder="Full Professional Name" required>
            <button type="submit" class="btn-main" style="background:#10b981">Provision Account</button>
            <span class="toggle-txt" onclick="toggle()">Already provisioned? Login</span>
        </form>
    </div>
    <script>
        function toggle() {
            document.getElementById('login-form').classList.toggle('hidden');
            document.getElementById('reg-form').classList.toggle('hidden');
            document.getElementById('form-title').innerText = document.getElementById('form-title').innerText.includes("Pro") ? "📝 Provision ID" : "🛡️ Fraud Shield Pro";
        }
    </script>
</body>
</html>
"""

DASHBOARD_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Fraud Shield Pro | AI Command Center</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root { --primary: #6366f1; --bg: #0b0f1a; --card: rgba(30, 41, 59, 0.4); }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: #f1f5f9; margin: 0; display: flex; height: 100vh; overflow: hidden; }
        
        /* Sidebar Navigation */
        .sidebar { width: 260px; background: #0f172a; border-right: 1px solid #1e293b; padding: 30px 20px; display: flex; flex-direction: column; gap: 20px; }
        .nav-item { padding: 12px 15px; border-radius: 10px; cursor: pointer; color: #94a3b8; display: flex; align-items: center; gap: 12px; transition: 0.2s; }
        .nav-item.active { background: var(--primary); color: white; }
        .nav-item:hover:not(.active) { background: #1e293b; color: white; }

        /* Main Content */
        .main { flex: 1; padding: 40px; overflow-y: auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: 380px 1fr; gap: 25px; }
        .card { background: var(--card); border: 1px solid rgba(255,255,255,0.1); padding: 25px; border-radius: 18px; backdrop-filter: blur(10px); }
        
        /* Form & Map Elements */
        label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; display: block; margin-top: 15px; }
        input, select { width: 100%; padding: 12px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; color: white; margin-top: 5px; }
        #map { height: 400px; border-radius: 15px; border: 1px solid #1e293b; }
        .ledger-table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }
        .ledger-table th { text-align: left; color: #64748b; padding: 10px; border-bottom: 1px solid #1e293b; }
        .ledger-table td { padding: 10px; border-bottom: 1px solid #1e293b; color: #cbd5e1; }
        .tag { padding: 4px 8px; border-radius: 5px; font-weight: bold; }
        .HIGH { color: #f87171; background: rgba(248, 113, 113, 0.1); }
        .LOW { color: #4ade80; background: rgba(74, 222, 128, 0.1); }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2 style="margin-top:0;">🛡️ SHIELD <span style="color:var(--primary)">PRO</span></h2>
        <div id="tab-audit" class="nav-item active" onclick="switchSection('audit')"><i class="fas fa-search-dollar"></i> Fraud Audit</div>
        <div id="tab-policy" class="nav-item" onclick="switchSection('policy')"><i class="fas fa-user-shield"></i> Policy Registry</div>
        <div style="margin-top:auto;"><a href="/logout" class="nav-item" style="text-decoration:none;"><i class="fas fa-sign-out-alt"></i> Sign Out</a></div>
    </div>

    <div class="main">
        <div class="header">
            <div>
                <h1 style="margin:0;">Analytics Dashboard</h1>
                <p style="color:#64748b; margin:5px 0 0 0;">AI Engine Status: <span style="color:#4ade80">● Operational</span></p>
            </div>
            <div style="text-align:right;">
                <span style="font-size:14px; color:#94a3b8;">Investigator ID: <b>{{ user_name }}</b></span>
            </div>
        </div>

        <div id="auditSection" class="section">
            <div class="grid">
                <div class="card">
                    <h3 style="margin-top:0;">Risk Entry</h3>
                    <label>Region (Kerala Hubs)</label>
                    <select id="Region">
                        <option value="Trivandrum">Trivandrum (South)</option>
                        <option value="Kochi">Kochi (Central)</option>
                        <option value="Kozhikode">Kozhikode (North)</option>
                        <option value="Munnar">Munnar (Highlands)</option>
                    </select>
                    <label>Claim Amount (INR)</label><input type="number" id="Amount" value="45000">
                    <label>Policy Duration (Days)</label><input type="number" id="Days" value="12">
                    <button onclick="performAudit()" style="margin-top:20px; background:var(--primary); padding:15px; border-radius:10px; border:none; color:white; font-weight:bold; cursor:pointer; width:100%;">Initiate AI Audit</button>
                    
                    <div style="margin-top:30px; text-align:center;">
                        <canvas id="gaugeChart" style="max-height:130px;"></canvas>
                        <h2 id="scoreDisplay" style="margin-bottom:0;">--%</h2>
                        <small style="color:#64748b;">Probability Index</small>
                    </div>
                </div>
                
                <div class="card">
                    <h3 style="margin-top:0;">Geo-Spatial Context</h3>
                    <div id="map"></div>
                    <div id="explanations" style="margin-top:15px; display:flex; gap:10px; flex-wrap:wrap;"></div>
                </div>
            </div>

            <div class="card" style="margin-top:25px;">
                <h3 style="margin-top:0;">Digital Audit Ledger (Blockchain-Lite)</h3>
                <table class="ledger-table">
                    <thead><tr><th>TIMESTAMP</th><th>LOCATION</th><th>SCORE</th><th>TAMPER-PROOF TOKEN</th></tr></thead>
                    <tbody id="ledgerBody"></tbody>
                </table>
            </div>
        </div>

        <div id="policySection" class="section" style="display:none;">
            <div class="card" style="max-width:500px;">
                <h3>Register Global Policy</h3>
                <label>Full Name</label><input type="text" id="pName" placeholder="John Doe">
                <label>Policy Category</label><select><option>Auto-Industrial</option><option>Property-Tech</option><option>Life-Premium</option></select>
                <button onclick="alert('Policy Hash Generated and Stored.')" style="margin-top:20px; background:#10b981; border:none; padding:12px; border-radius:8px; color:white; font-weight:bold; cursor:pointer; width:100%;">Generate Contract</button>
            </div>
        </div>
    </div>

    <script>
        let gauge, map, marker;
        const keralaCoords = { 'Trivandrum': [8.5241, 76.9366], 'Kochi': [9.9312, 76.2673], 'Kozhikode': [11.2588, 75.7804], 'Munnar': [10.0889, 77.0595] };

        window.onload = () => {
            const ctx = document.getElementById('gaugeChart').getContext('2d');
            gauge = new Chart(ctx, { type: 'doughnut', data: { datasets: [{ data: [0, 100], backgroundColor: ['#6366f1', '#1e293b'], borderWidth: 0, circumference: 180, rotation: 270, cutout: '85%' }] }, options: { animation: { duration: 1500 } } });

            map = L.map('map').setView([10.8505, 76.2711], 7);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);
        };

        function switchSection(sec) {
            document.querySelectorAll('.section').forEach(s => s.style.display = 'none');
            document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
            document.getElementById(sec + 'Section').style.display = 'block';
            document.getElementById('tab-' + sec).classList.add('active');
        }

        async function performAudit() {
            const region = document.getElementById('Region').value;
            const payload = { Age: 30, Claim_Amount: parseFloat(document.getElementById('Amount').value), Policy_Type: 'Auto', Days_Since_Purchase: parseInt(document.getElementById('Days').value), Region: region };

            const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await res.json();

            gauge.data.datasets[0].data = [data.score, 100 - data.score];
            gauge.data.datasets[0].backgroundColor = [data.score > 70 ? '#ef4444' : '#22c55e', '#1e293b'];
            gauge.update();

            document.getElementById('scoreDisplay').innerText = data.score + "%";
            document.getElementById('explanations').innerHTML = data.reasons.map(r => `<span style="background:rgba(99,102,241,0.1); border:1px solid #6366f1; padding:4px 10px; border-radius:15px; font-size:11px;">${r}</span>`).join('');

            // Map Update
            if (marker) map.removeLayer(marker);
            marker = L.circleMarker(keralaCoords[region], { color: data.score > 70 ? '#ef4444' : '#22c55e', radius: 12, fillOpacity: 0.7 }).addTo(map).bindPopup(`Audit Risk: ${data.score}%`).openPopup();
            map.flyTo(keralaCoords[region], 11);

            // Ledger Update
            const row = `<tr><td>${new Date().toLocaleTimeString()}</td><td>${region}</td><td class="${data.score > 70 ? 'HIGH' : 'LOW'}">${data.score}%</td><td style="font-family:monospace; color:#64748b;">${data.token.substring(0,25)}...</td></tr>`;
            document.getElementById('ledgerBody').insertAdjacentHTML('afterbegin', row);
        }
    </script>
</body>
</html>
"""

# --- PART 3: SERVER ARCHITECTURE ---

@app.route('/')
def home(): return render_template_string(LOGIN_UI)

@app.route('/auth', methods=['POST'])
def auth():
    u, p = request.form.get('u'), request.form.get('p')
    if u in user_db and user_db[u]['password'] == p:
        session['logged_in'], session['user_name'] = True, user_db[u]['name']
        return redirect(url_for('dashboard'))
    return redirect('/')

@app.route('/register', methods=['POST'])
def register():
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
    
    # Audit Logic
    token = hashlib.sha256(f"{data}-{score}-{datetime.now()}".encode()).hexdigest()
    reasons = ["🚩 Outlier Amount" if data['Claim_Amount'] > 25000 else "⚖️ Standard Value"]
    if data['Days_Since_Purchase'] < 20: reasons.append("⏱️ Rapid Liability")
    
    return jsonify({'score': score, 'reasons': reasons, 'token': token})

@app.route('/logout')
def logout():
    session.clear(); return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)