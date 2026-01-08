import pandas as pd
import numpy as np
import hashlib
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
app.secret_key = "shield_kerala_geo_v9"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///shield_v9.db'
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

class AuditLedger(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    region = db.Column(db.String(50))
    amount = db.Column(db.Float)
    score = db.Column(db.Float)
    token = db.Column(db.String(100))

# --- 3. AI ENGINE (District-Aware Training) ---
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

# --- 4. ADVANCED UI ASSETS ---
HEADER = """
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap" rel="stylesheet">
<style>
    body { font-family: 'Plus Jakarta Sans', sans-serif; background: #020617; color: #f8fafc; }
    .glass { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.05); }
    #map { height: 400px; border-radius: 2rem; border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 0 30px rgba(99, 102, 241, 0.1); }
    .leaflet-container { background: #020617 !important; }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
</style>
"""

LOGIN_UI = f"<!DOCTYPE html><html><head>{HEADER}</head><body>" + """
<div class="min-h-screen flex items-center justify-center p-6 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-slate-900 via-gray-950 to-black">
    <div class="glass max-w-lg w-full rounded-[3rem] p-12 shadow-2xl">
        <div class="text-center mb-10"><h1 class="text-5xl font-black tracking-tighter">SHIELD<span class="text-indigo-500">PRO</span></h1><p class="text-slate-500 mt-2">Geospatial Fraud Surveillance</p></div>
        {% with messages = get_flashed_messages() %}{% if messages %}{% for m in messages %}<div class="bg-red-500/10 text-red-400 p-4 rounded-xl text-xs mb-6 text-center border border-red-500/20">{{m}}</div>{% endfor %}{% endif %}{% endwith %}
        <div class="flex p-1.5 bg-black/40 rounded-2xl mb-8"><button onclick="tab('login')" id="l-btn" class="flex-1 py-3 rounded-xl text-sm font-bold bg-indigo-600 text-white">Sign In</button><button onclick="tab('reg')" id="r-btn" class="flex-1 py-3 rounded-xl text-sm font-bold text-slate-500">Register</button></div>
        <form action="/auth" method="POST" id="authForm" class="space-y-4">
            <input type="hidden" name="action" id="action" value="login">
            <div id="reg-fields" class="hidden space-y-4">
                <input name="full_name" placeholder="Full Name" class="w-full bg-slate-900/50 border border-white/10 rounded-2xl px-6 py-4 outline-none">
                <div class="grid grid-cols-2 gap-4"><input name="email" type="email" placeholder="Email" class="w-full bg-slate-900/50 border border-white/10 rounded-2xl px-6 py-4 outline-none"><input name="age" type="number" placeholder="Age" class="w-full bg-slate-900/50 border border-white/10 rounded-2xl px-6 py-4 outline-none"></div>
            </div>
            <input name="username" placeholder="Username" required class="w-full bg-slate-900/50 border border-white/10 rounded-2xl px-6 py-4 outline-none focus:border-indigo-500">
            <input name="password" id="p1" type="password" placeholder="Password" required class="w-full bg-slate-900/50 border border-white/10 rounded-2xl px-6 py-4 outline-none focus:border-indigo-500">
            <div id="conf-field" class="hidden"><input id="p2" type="password" placeholder="Confirm Password" class="w-full bg-slate-900/50 border border-white/10 rounded-2xl px-6 py-4 outline-none"></div>
            <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 py-5 rounded-2xl font-black text-white shadow-xl shadow-indigo-600/20 transition-all active:scale-95">Enter Command Center</button>
        </form>
    </div>
</div>
<script>
    function tab(mode) {
        const isR = mode === 'reg';
        document.getElementById('action').value = isR ? 'register' : 'login';
        document.getElementById('reg-fields').classList.toggle('hidden', !isR);
        document.getElementById('conf-field').classList.toggle('hidden', !isR);
        document.getElementById('l-btn').className = isR ? 'flex-1 py-3 text-slate-500' : 'flex-1 py-3 bg-indigo-600 text-white rounded-xl font-bold';
        document.getElementById('r-btn').className = !isR ? 'flex-1 py-3 text-slate-500' : 'flex-1 py-3 bg-indigo-600 text-white rounded-xl font-bold';
    }
</script>
</body></html>
"""

DASHBOARD_UI = f"<!DOCTYPE html><html><head>{HEADER}</head><body>" + """
<div class="flex h-screen overflow-hidden">
    <aside class="w-72 glass border-r border-white/5 flex flex-col p-8">
        <h2 class="text-2xl font-black text-indigo-500 mb-12">SHIELD<span class="text-white">PRO</span></h2>
        <div class="bg-indigo-600/5 p-6 rounded-3xl border border-indigo-500/10 mb-8 text-center">
            <div class="w-12 h-12 bg-indigo-600 rounded-2xl mx-auto mb-3 flex items-center justify-center font-bold">{{ user.full_name[0] }}</div>
            <h4 class="font-bold text-xs">{{ user.full_name }}</h4>
        </div>
        <nav class="flex-1 space-y-2 text-sm font-bold text-slate-400">
            <div class="p-4 bg-white/5 text-indigo-400 rounded-2xl">📍 Kerala Map View</div>
            <div class="p-4 hover:bg-white/5 rounded-2xl transition-all cursor-pointer">📊 Advanced Stats</div>
        </nav>
        <a href="/logout" class="bg-red-500/10 text-red-500 p-4 rounded-2xl text-center text-xs font-bold border border-red-500/10 hover:bg-red-500 hover:text-white transition-all">Logout Session</a>
    </aside>

    <main class="flex-1 p-10 overflow-y-auto">
        <div class="grid grid-cols-12 gap-8">
            <div class="col-span-12 flex justify-between items-end">
                <div><h1 class="text-4xl font-black">Audit Surveillance</h1><p class="text-slate-500">Processing live claims across Kerala Districts</p></div>
                <div class="text-right text-[10px] font-black tracking-widest text-indigo-400 uppercase">System Time: {{ now }}</div>
            </div>

            <div class="col-span-8 space-y-8">
                <div class="glass p-4 rounded-[2.5rem]"><div id="map"></div></div>
                <div class="glass rounded-[2.5rem] p-10">
                    <h3 class="font-bold mb-6 text-xl">Risk Trajectory (Real-time)</h3>
                    <canvas id="liveChart" height="120"></canvas>
                </div>
            </div>

            <div class="col-span-4 space-y-8">
                <div class="glass rounded-[2.5rem] p-10">
                    <h3 class="font-bold mb-6 text-xl">Neural Entry</h3>
                    <div class="space-y-4">
                        <label class="text-[10px] font-bold text-slate-500 uppercase">Target District</label>
                        <select id="Region" class="w-full bg-slate-950 border border-white/10 rounded-2xl p-4 outline-none focus:border-indigo-500">
                            {% for r in kerala_regions %}<option value="{{r}}">{{r}}</option>{% endfor %}
                        </select>
                        <label class="text-[10px] font-bold text-slate-500 uppercase">Claim Amount</label>
                        <input type="number" id="Amount" value="55000" class="w-full bg-slate-950 border border-white/10 rounded-2xl p-4 outline-none">
                        <button onclick="runAudit()" class="w-full bg-indigo-600 py-5 rounded-2xl font-black shadow-lg shadow-indigo-600/20 active:scale-95 transition-all">Execute AI Audit</button>
                    </div>
                </div>
                <div class="glass rounded-[2.5rem] p-8 max-h-[450px] overflow-y-auto">
                    <h3 class="font-bold mb-4 text-slate-500 text-xs uppercase tracking-widest">Audit History</h3>
                    <div id="ledger" class="space-y-4">
                        {% for entry in history %}
                        <div class="bg-white/5 p-4 rounded-2xl border border-white/5">
                            <div class="flex justify-between font-black text-xs mb-1"><span>{{entry.region}}</span><span class="{{ 'text-red-400' if entry.score > 70 else 'text-green-400' }}">{{entry.score}}%</span></div>
                            <div class="text-[9px] font-mono text-slate-600 truncate">{{entry.token}}</div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
    </main>
</div>

<script>
    // Detailed Kerala Coordinates
    const keralaCoords = {
        'Trivandrum': [8.5241, 76.9366],
        'Kochi': [9.9312, 76.2673],
        'Kozhikode': [11.2588, 75.7804],
        'Munnar': [10.0889, 77.0595],
        'Wayanad': [11.6854, 76.1320],
        'Alappuzha': [9.4981, 76.3388],
        'Thrissur': [10.5276, 76.2144],
        'Palakkad': [10.7867, 76.6547],
        'Kannur': [11.8745, 75.3704],
        'Kollam': [8.8932, 76.6141]
    };

    let map = L.map('map').setView([10.5, 76.5], 7);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);

    const ctx = document.getElementById('liveChart').getContext('2d');
    let liveChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Risk %', data: [], borderColor: '#6366f1', tension: 0.4, fill: true, backgroundColor: 'rgba(99,102,241,0.05)' }] },
        options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.05)' } } } }
    });

    async function runAudit() {
        const region = document.getElementById('Region').value;
        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                Age: 32, Claim_Amount: parseFloat(document.getElementById('Amount').value),
                Policy_Type: 'Auto', Days_Since_Purchase: 12, Region: region
            })
        });
        const d = await res.json();
        
        // Map Animation & Marker
        map.flyTo(keralaCoords[region], 10);
        L.circleMarker(keralaCoords[region], {
            radius: 15, color: d.score > 70 ? '#f87171' : '#4ade80',
            fillColor: d.score > 70 ? '#ef4444' : '#22c55e', fillOpacity: 0.5
        }).addTo(map).bindPopup(`<b>${region} Risk: ${d.score}%</b>`).openPopup();

        // UI Updates
        const entry = `<div class="bg-indigo-600/10 p-4 rounded-2xl border border-indigo-500/20"><div class="flex justify-between font-black text-xs"><span>${region}</span><span class="text-indigo-400">${d.score}%</span></div></div>`;
        document.getElementById('ledger').insertAdjacentHTML('afterbegin', entry);
        liveChart.data.labels.push(new Date().toLocaleTimeString());
        liveChart.data.datasets[0].data.push(d.score);
        liveChart.update();
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
        flash("Registration Success!"); return redirect(url_for('index'))
    user = User.query.filter_by(username=u).first()
    if user and bcrypt.check_password_hash(user.password, p):
        session['user_id'] = user.id; return redirect(url_for('dashboard'))
    flash("Invalid Login!"); return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect(url_for('index'))
    user = User.query.get(session['user_id'])
    history = AuditLedger.query.order_by(AuditLedger.timestamp.desc()).all()
    return render_template_string(DASHBOARD_UI, user=user, history=history, kerala_regions=KERALA_REGIONS, now=datetime.now().strftime("%H:%M:%S"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    proc = PREPROCESSOR.transform(pd.DataFrame([data]))
    score = round(float(MODEL.predict_proba(proc)[:, 1][0]) * 100, 1)
    token = hashlib.sha256(f"{data}-{score}".encode()).hexdigest()
    db.session.add(AuditLedger(region=data['Region'], amount=data['Claim_Amount'], score=score, token=token))
    db.session.commit()
    return jsonify({'score': score, 'token': token})

@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    app.run(debug=True)