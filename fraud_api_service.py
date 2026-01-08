import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'fraud_model.pkl')
PREPROCESSOR_FILE = os.path.join(BASE_DIR, 'preprocessor.pkl')

# --- DATA STORAGE (In-Memory for this demo) ---
registered_policies = []

# --- PART 1: AI MODEL LOGIC ---
def train_model():
    print("Training AI Engine...")
    np.random.seed(42)
    n = 5000
    data = {
        'Age': np.random.randint(18, 70, n),
        'Claim_Amount': np.random.uniform(500, 25000, n),
        'Policy_Type': np.random.choice(['Auto', 'Home', 'Life'], n),
        'Days_Since_Purchase': np.random.randint(1, 1000, n),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'Fraud_Reported': np.random.choice([0, 1], n, p=[0.92, 0.08])
    }
    df = pd.DataFrame(data)
    X = df.drop('Fraud_Reported', axis=1)
    y = df['Fraud_Reported']
    ct = ColumnTransformer([
        ('num', StandardScaler(), ['Age', 'Claim_Amount', 'Days_Since_Purchase']),
        ('cat', OneHotEncoder(), ['Policy_Type', 'Region'])
    ])
    X_processed = ct.fit_transform(X)
    X_res, y_res = SMOTE().fit_resample(X_processed, y)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss')
    model.fit(X_res, y_res)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(ct, PREPROCESSOR_FILE)
    return model, ct

if not os.path.exists(MODEL_FILE):
    MODEL, PREPROCESSOR = train_model()
else:
    MODEL, PREPROCESSOR = joblib.load(MODEL_FILE), joblib.load(PREPROCESSOR_FILE)

# --- PART 2: THE MODERN UI ---
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Shield AI | Policy & Claims</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root { --primary: #6366f1; --bg: #0f172a; --card-bg: rgba(30, 41, 59, 0.7); }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: white; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        
        .dashboard { 
            background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1);
            padding: 30px; border-radius: 24px; width: 1100px; display: grid; grid-template-columns: 1fr 1fr; gap: 30px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); animation: slideUp 0.8s ease-out;
        }

        @keyframes slideUp { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: translateY(0); } }

        .tabs { grid-column: 1 / span 2; display: flex; gap: 20px; border-bottom: 1px solid #334155; padding-bottom: 10px; }
        .tab { cursor: pointer; padding: 10px 20px; border-radius: 8px; transition: 0.3s; color: #94a3b8; }
        .tab.active { background: var(--primary); color: white; }

        .section { display: none; }
        .section.active { display: block; }

        .form-group { margin-bottom: 15px; }
        label { display: block; font-size: 12px; color: #94a3b8; margin-bottom: 5px; }
        input, select { 
            width: 100%; padding: 10px; background: rgba(15, 23, 42, 0.5); border: 1px solid #334155; 
            border-radius: 8px; color: white; 
        }

        button { 
            width: 100%; padding: 12px; background: var(--primary); 
            border: none; border-radius: 10px; color: white; font-weight: bold; cursor: pointer; margin-top: 10px;
        }

        .viz-box { background: rgba(15, 23, 42, 0.3); padding: 20px; border-radius: 20px; text-align: center; }
        #gaugeChart { max-width: 250px; margin: auto; }
        
        .status-badge { padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: bold; }
        .HIGH { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
        .LOW { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }

        .counter { font-size: 24px; font-weight: bold; color: var(--primary); }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="tabs">
            <div class="tab active" onclick="switchTab('audit')"><i class="fas fa-search"></i> Fraud Audit</div>
            <div class="tab" onclick="switchTab('policy')"><i class="fas fa-plus-circle"></i> New Policy</div>
        </div>

        <div>
            <div id="auditSection" class="section active">
                <h3 style="color:#818cf8">Claim Audit Engine</h3>
                <div class="form-group">
                    <label>Select Registered Policy</label>
                    <select id="policySelect" onchange="loadPolicyData()"><option value="">Manual Entry...</option></select>
                </div>
                <div class="form-group"><label>Age</label><input type="number" id="Age" value="30"></div>
                <div class="form-group"><label>Claim Amount ($)</label><input type="number" id="Claim_Amount" value="18000"></div>
                <div class="form-group"><label>Policy Type</label><select id="Policy_Type"><option>Auto</option><option>Home</option><option>Life</option></select></div>
                <div class="form-group"><label>Days Since Purchase</label><input type="number" id="Days_Since_Purchase" value="15"></div>
                <div class="form-group"><label>Region</label><select id="Region"><option>North</option><option>South</option><option>East</option><option>West</option></select></div>
                <button onclick="analyze()">Perform Risk Audit</button>
            </div>

            <div id="policySection" class="section">
                <h3 style="color:#10b981">Policy Registration</h3>
                <div class="form-group"><label>Policy Holder Name</label><input type="text" id="pName" placeholder="e.g. John Doe"></div>
                <div class="form-group"><label>Holder Age</label><input type="number" id="pAge" value="30"></div>
                <div class="form-group"><label>Initial Coverage ($)</label><input type="number" id="pCover" value="5000"></div>
                <div class="form-group"><label>Policy Category</label><select id="pType"><option>Auto</option><option>Home</option><option>Life</option></select></div>
                <div class="form-group"><label>Region</label><select id="pRegion"><option>North</option><option>South</option><option>East</option><option>West</option></select></div>
                <button style="background:#10b981" onclick="registerPolicy()">Register Policy</button>
            </div>
        </div>

        <div class="viz-box">
            <div class="counter" id="policyCount">0</div>
            <div style="font-size:12px; color:#94a3b8; margin-bottom:20px;">Live Registered Policies</div>
            <canvas id="gaugeChart"></canvas>
            <div id="riskLevel" style="font-size:24px; font-weight:bold; margin-top:20px;">Ready</div>
            <p id="riskDetail" style="color:#94a3b8; font-size:14px;"></p>
        </div>

        <div style="grid-column: 1 / span 2; margin-top:20px;">
            <table style="width:100%; border-top:1px solid #334155;">
                <thead><tr style="color:#64748b; font-size:12px;"><th>Holder</th><th>Amount</th><th>Score</th><th>Result</th></tr></thead>
                <tbody id="historyBody"></tbody>
            </table>
        </div>
    </div>

    <script>
        let gauge;
        let policies = [];

        window.onload = () => {
            const ctx = document.getElementById('gaugeChart').getContext('2d');
            gauge = new Chart(ctx, {
                type: 'doughnut',
                data: { datasets: [{ data: [0, 100], backgroundColor: ['#6366f1', '#1e293b'], borderWidth: 0, circumference: 180, rotation: 270, cutout: '85%' }] },
                options: { animation: { duration: 1500 } }
            });
        };

        function switchTab(tab) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tab + 'Section').classList.add('active');
            event.currentTarget.classList.add('active');
        }

        function registerPolicy() {
            const p = {
                name: document.getElementById('pName').value,
                age: document.getElementById('pAge').value,
                type: document.getElementById('pType').value,
                region: document.getElementById('pRegion').value,
                date: new Date()
            };
            policies.push(p);
            updatePolicyUI();
            switchTab('audit');
            alert("Policy Registered Successfully!");
        }

        function updatePolicyUI() {
            document.getElementById('policyCount').innerText = policies.length;
            const select = document.getElementById('policySelect');
            select.innerHTML = '<option value="">Manual Entry...</option>' + 
                policies.map((p, i) => `<option value="${i}">${p.name} (${p.type})</option>`).join('');
        }

        function loadPolicyData() {
            const idx = document.getElementById('policySelect').value;
            if(idx !== "") {
                const p = policies[idx];
                document.getElementById('Age').value = p.age;
                document.getElementById('Policy_Type').value = p.type;
                document.getElementById('Region').value = p.region;
                // Auto-calculate days since purchase (simplified)
                document.getElementById('Days_Since_Purchase').value = 1; 
            }
        }

        async function analyze() {
            const payload = {
                Age: parseInt(document.getElementById('Age').value),
                Claim_Amount: parseFloat(document.getElementById('Claim_Amount').value),
                Policy_Type: document.getElementById('Policy_Type').value,
                Days_Since_Purchase: parseInt(document.getElementById('Days_Since_Purchase').value),
                Region: document.getElementById('Region').value
            };

            const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await res.json();

            gauge.data.datasets[0].data = [data.score, 100 - data.score];
            gauge.data.datasets[0].backgroundColor = [data.score > 70 ? '#ef4444' : '#22c55e', '#1e293b'];
            gauge.update();

            document.getElementById('riskLevel').innerText = data.score + "% RISK";
            document.getElementById('riskLevel').style.color = data.score > 70 ? '#f87171' : '#4ade80';
            document.getElementById('riskDetail').innerText = data.reasons[0];

            const holder = document.getElementById('policySelect').options[document.getElementById('policySelect').selectedIndex].text.split(' (')[0];
            const row = `<tr><td>${holder}</td><td>$${payload.Claim_Amount}</td><td>${data.score}%</td><td><span class="status-badge ${data.level}">${data.level}</span></td></tr>`;
            document.getElementById('historyBody').insertAdjacentHTML('afterbegin', row);
        }
    </script>
</body>
</html>
"""

# --- PART 3: API ROUTES ---
@app.route('/')
def index(): return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    processed = PREPROCESSOR.transform(pd.DataFrame([data]))
    prob = float(MODEL.predict_proba(processed)[:, 1][0])
    score = round(prob * 100, 1)
    
    reasons = []
    if data['Claim_Amount'] > 18000: reasons.append("Flagged: Extreme claim value detected")
    elif data['Days_Since_Purchase'] < 30: reasons.append("Flagged: Immediate claim after purchase")
    else: reasons.append("Verified: Normal behavioral pattern")

    return jsonify({
        'score': score,
        'level': "HIGH" if score > 70 else "MEDIUM" if score > 40 else "LOW",
        'reasons': reasons
    })

if __name__ == '__main__':
    app.run(debug=True)