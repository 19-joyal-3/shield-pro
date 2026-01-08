import pandas as pd
import numpy as np
import hashlib
import hmac
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# --- 1. SYSTEM CONFIGURATION ---
app = Flask(__name__)
app.secret_key = "shield_kerala_integrity_v10" 
# This key is used for the Safety Value (HMAC)
INTEGRITY_SECRET = b'kerala_audit_integrity_key_2026'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///shield_v10.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# --- 2. DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100))
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AuditLedger(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    region = db.Column(db.String(50))
    amount = db.Column(db.Float)
    score = db.Column(db.Float)
    integrity_hash = db.Column(db.String(128)) # Stores the Safety Value

# --- 3. AI ENGINE ---
def train_enterprise_model():
    np.random.seed(42); n = 1500
    regions = ['Trivandrum', 'Kochi', 'Kozhikode', 'Munnar', 'Wayanad', 'Alappuzha', 'Thrissur', 'Palakkad', 'Kannur', 'Kollam']
    data = {
        'Age': np.random.randint(18, 75, n),
        'Claim_Amount': np.random.uniform(500, 60000, n),
        'Policy_Type': np.random.choice(['Auto', 'Property', 'Health', 'Travel'], n),
        'Days_Since_Purchase': np.random.randint(1, 1000, n),
        'Region': np.random.choice(regions, n),
        'Fraud_Reported': np.random.choice([0, 1], n, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)
    ct = ColumnTransformer([
        ('num', StandardScaler(), [0, 1, 3]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), [2, 4])
    ])
    X_processed = ct.fit_transform(df.drop('Fraud_Reported', axis=1))
    X_res, y_res = SMOTE().fit_resample(X_processed, df['Fraud_Reported'])
    model = xgb.XGBClassifier(eval_metric='logloss').fit(X_res, y_res)
    return model, ct, regions

MODEL, PREPROCESSOR, KERALA_REGIONS = train_enterprise_model()

# --- 4. UI TEMPLATES ---
HEADER = """
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap" rel="stylesheet">
<style>
    body { font-family: 'Plus Jakarta Sans', sans-serif; background: #010409; color: #f0f6fc; }
    .glass { background: rgba(22, 27, 34, 0.8); backdrop-filter: blur(12px); border: 1px solid #30363d; }
    #map { height: 400px; border-radius: 1.5rem; border: 1px solid #30363d; }
    .safety-tag { font-family: 'Courier New', monospace; font-size: 0.65rem; color: #8b949e; }
</style>
"""

LOGIN_UI = f"<!DOCTYPE html><html><head>{HEADER}</head><body>" + """
<div class="min-h-screen flex items-center justify-center p-6 bg-slate-950">
    <div class="glass max-w-lg w-full rounded-3xl p-10 shadow-2xl">
        <div class="text-center mb-10"><h1 class="text-4xl font-black">SHIELD<span class="text-indigo-500">PRO</span></h1><p class="text-slate-500 mt-2 text-sm uppercase tracking-widest font-bold">Secure Audit Vault</p></div>
        {% with messages = get_flashed_messages() %}{% if messages %}{% for m in messages %}<div class="bg-indigo-500/10 text-indigo-400 p-3 rounded-xl text-xs mb-6 text-center border border-indigo-500/20">{{m}}</div>{% endfor %}{% endif %}{% endwith %}
        <div class="flex p-1.5 bg-black rounded-2xl mb-8 border border-slate-800"><button onclick="tab('login')" id="l-btn" class="flex-1 py-3 rounded-xl text-sm font-bold bg-indigo-600 text-white">Sign In</button><button onclick="tab('reg')" id="r-btn" class="flex-1 py-3 rounded-xl text-sm font-bold text-slate-500">Register</button></div>
        <form action="/auth" method="POST" class="space-y-4">
            <input type="hidden" name="action" id="action" value="login">
            <div id="reg-fields" class="hidden space-y-4">
                <input name="full_name" placeholder="Full Name" class="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-indigo-600">
                <div class="grid grid-cols-2 gap-4"><input name="email" type="email" placeholder="Email" class="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 outline-none"><input name="age" type="number" placeholder="Age" class="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 outline-none"></div>
            </div>
            <input name="username" placeholder="Username" required class="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 outline-none">
            <input name="password" type="password" placeholder="Password" required class="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 outline-none">
            <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 py-4 rounded-xl font-black transition-all shadow-lg shadow-indigo-600/20">Access Vault</button>
        </form>
    </div>
</div>
<script>
    function tab(mode) {
        const isR = mode === 'reg';
        document.getElementById('action').value = isR ? 'register' : 'login';
        document.getElementById('reg-fields').classList.toggle('hidden', !isR);
        document.getElementById('l-btn').className = isR ? 'flex-1 py-3 text-slate-500' : 'flex-1 py-3 bg-indigo-600 text-white rounded-xl font-bold';
        document.getElementById('r-btn').className = !isR ? 'flex-1 py-3 text-slate-500' : 'flex-1 py-3 bg-indigo-600 text-white rounded-xl font-bold';
    }
</script>
</body></html>
"""

DASHBOARD_UI = f"<!DOCTYPE html><html><head>{HEADER}</head><body>" + """
<div class="flex h-screen overflow-hidden">
    <aside class="w-72 glass border-r border-slate-800 flex flex-col p-8">
        <h2 class="text-2xl font-black text-indigo-500 mb-8">SHIELD PRO</h2>
        <div class="bg-indigo-600/10 p-5 rounded-2xl border border-indigo-500/20 mb-8">
            <p class="text-[10px] uppercase font-black text-indigo-400 tracking-widest mb-1">Identity Verified</p>
            <h4 class="font-bold text-sm">{{ user.full_name }}</h4>
            <p class="text-[10px] text-slate-500 truncate">{{ user.email }}</p>
            <p class="text-[10px] text-slate-500">Age: {{ user.age }}</p>
        </div>
        <nav class="flex-1 space-y-2 text-xs font-bold text-slate-400 uppercase tracking-wider">
            <div class="p-4 bg-slate-800 text-indigo-400 rounded-xl">📍 Kerala Audit Map</div>
        </nav>
        <a href="/logout" class="bg-red-500/10 text-red-500 p-4 rounded-xl text-center text-xs font-bold border border-red-500/10 hover:bg-red-500 transition-all">Logout Session</a>
    </aside>

    <main class="flex-1 p-10 overflow-y-auto">
        <div class="grid grid-cols-12 gap-6">
            <div class="col-span-12 mb-4 flex justify-between items-end">
                <div><h1 class="text-3xl font-black">Safety Ledger & Intelligence</h1><p class="text-slate-500 text-sm">Hashed Integrity Verification Active</p></div>
            </div>

            <div class="col-span-8 space-y-6">
                <div class="glass p-4 rounded-3xl"><div id="map"></div></div>
                <div class="glass rounded-3xl p-8">
                    <canvas id="liveChart" height="100"></canvas>
                </div>
            </div>

            <div class="col-span-4 space-y-6">
                <div class="glass rounded-3xl p-8">
                    <h3 class="font-bold mb-6 text-indigo-400 text-xs uppercase tracking-widest">New Verification</h3>
                    <div class="space-y-4">
                        <select id="Region" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 outline-none">
                            {% for r in kerala_regions %}<option value="{{r}}">{{r}}</option>{% endfor %}
                        </select>
                        <input type="number" id="Amount" value="50000" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 outline-none">
                        <button onclick="runAudit()" class="w-full bg-indigo-600 py-3 rounded-xl font-bold active:scale-95 transition-all">Generate Safety Score</button>
                    </div>
                </div>
                <div class="glass rounded-3xl p-6 max-h-[400px] overflow-y-auto">
                    <h3 class="font-bold mb-4 text-slate-500 text-[10px] uppercase">Safety Verified Ledger</h3>
                    <div id="ledger" class="space-y-4">
                        {% for entry in history %}
                        <div class="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                            <div class="flex justify-between font-black text-xs mb-2"><span>{{entry.region}}</span><span class="{{ 'text-red-400' if entry.score > 70 else 'text-green-400' }}">{{entry.score}}%</span></div>
                            <div class="safety-tag truncate">MAC: {{entry.integrity_hash}}</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
    </main>
</div>
<script>
    const keralaCoords = { 'Trivandrum': [8.5241, 76.9366], 'Kochi': [9.9312, 76.2673], 'Kozhikode': [11.2588, 75.7804], 'Munnar': [10.0889, 77.0595], 'Wayanad': [11.6854, 76.1320], 'Alappuzha': [9.4981, 76.3388], 'Thrissur': [10.5276, 76.2144], 'Palakkad': [10.7867, 76.6547], 'Kannur': [11.8745, 75.3704], 'Kollam': [8.8932, 76.6141] };
    let map = L.map('map').setView([10.5, 76.5], 7);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);
    const ctx = document.getElementById('liveChart').getContext('2d');
    let chart = new Chart(ctx, { type: 'line', data: { labels: [], datasets: [{ label: 'Safety Trend', data: [], borderColor: '#6366f1', tension: 0.4 }] }, options: { scales: { y: { beginAtZero: true, max: 100 } } } });

    async function runAudit() {
        const r = document.getElementById('Region').value;
        const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ Age: 30, Claim_Amount: parseFloat(document.getElementById('Amount').value), Policy_Type: 'Auto', Days_Since_Purchase: 10, Region: r }) });
        const d = await res.json();
        map.flyTo(keralaCoords[r], 11);
        L.circleMarker(keralaCoords[r], { radius: 10, color: '#6366f1' }).addTo(map).bindPopup(`${r}: ${d.score}%`).openPopup();
        const html = `<div class="bg-indigo-600/10 p-4 rounded-xl border border-indigo-500/20"><div class="flex justify-between font-black text-xs"><span>${r}</span><span>${d.score}%</span></div><div class="safety-tag truncate">MAC: ${d.safety_value}</div></div>`;
        document.getElementById('ledger').insertAdjacentHTML('afterbegin', html);
        chart.data.labels.push(new Date().toLocaleTimeString()); chart.data.datasets[0].data.push(d.score); chart.update();
    }
</script>
</body></html>
"""

# --- 5. SERVER ROUTES ---
@app.route('/')
def index():
    return render_template_string(LOGIN_UI)

@app.route('/auth', methods=['POST'])
def auth():
    action = request.form.get('action')
    u, p = request.form.get('username'), request.form.get('password')
    if action == 'register':
        if User.query.filter_by(username=u).first(): flash("User exists!"); return redirect(url_for('index'))
        age = request.form.get('age', '0')
        age_val = int(age) if (age and age.isdigit()) else 0
        hashed = bcrypt.generate_password_hash(p).decode('utf-8')
        db.session.add(User(username=u, password=hashed, full_name=request.form.get('full_name'), email=request.form.get('email'), age=age_val))
        db.session.commit()
        flash("Vault Profile Created!"); return redirect(url_for('index'))
    user = User.query.filter_by(username=u).first()
    if user and bcrypt.check_password_hash(user.password, p):
        session['user_id'] = user.id; return redirect(url_for('dashboard'))
    flash("Authentication Denied!"); return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect(url_for('index'))
    user = User.query.get(session['user_id'])
    history = AuditLedger.query.order_by(AuditLedger.timestamp.desc()).all()
    return render_template_string(DASHBOARD_UI, user=user, history=history, kerala_regions=KERALA_REGIONS)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    proc = PREPROCESSOR.transform(pd.DataFrame([data]))
    score = round(float(MODEL.predict_proba(proc)[:, 1][0]) * 100, 1)
    
    # GENERATE SAFETY VALUE (HMAC-SHA256 Signature)
    # This prevents anyone from manually changing the data in the database
    msg = f"{data['Region']}-{data['Claim_Amount']}-{score}-{datetime.now().strftime('%Y%m%d')}"
    safety_value = hmac.new(INTEGRITY_SECRET, msg.encode(), hashlib.sha256).hexdigest()
    
    db.session.add(AuditLedger(region=data['Region'], amount=data['Claim_Amount'], score=score, integrity_hash=safety_value))
    db.session.commit()
    return jsonify({'score': score, 'safety_value': safety_value})

@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    app.run(debug=True)