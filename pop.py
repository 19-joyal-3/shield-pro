import pandas as pd
import numpy as np
import hashlib
import hmac
import io
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sqlalchemy.exc import IntegrityError # Added for error handling
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# AI & Data Libraries
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    from twilio.rest import Client
    HAS_TWILIO = True
except ImportError:
    HAS_TWILIO = False

import xgboost as xgb
from fpdf import FPDF

# --- 1. SYSTEM CONFIGURATION ---
app = Flask(__name__)
app.secret_key = "shield_v28_master_ultimate_kerala" 
INTEGRITY_SECRET = b'kerala_audit_integrity_key_2026'

# Credentials Configuration
TWILIO_SID = 'AC_YOUR_SID'
TWILIO_TOKEN = 'YOUR_TOKEN'
TWILIO_PHONE = '+1234567890'
SENDER_EMAIL = "your-email@gmail.com"
SENDER_PASSWORD = "your-app-password"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///shield_master_v28.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# --- 2. DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100))
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20)) 
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='Auditor') 

class AuditLedger(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    claimant_name = db.Column(db.String(100))
    claimant_age = db.Column(db.Integer)
    region = db.Column(db.String(50))
    amount = db.Column(db.Float)          
    coverage_limit = db.Column(db.Float)   
    tenure_months = db.Column(db.Integer)  
    score = db.Column(db.Float)
    integrity_hash = db.Column(db.String(128))
    auditor_id = db.Column(db.Integer, db.ForeignKey('user.id'))

# --- 3. AI ENGINE (KERALA TUNED) ---
def train_enterprise_model():
    kerala_districts = ['Thiruvananthapuram', 'Kollam', 'Pathanamthitta', 'Alappuzha', 'Kottayam', 'Idukki', 'Ernakulam', 'Thrissur', 'Palakkad', 'Malappuram', 'Kozhikode', 'Wayanad', 'Kannur', 'Kasaragod']
    
    # Check if local dataset exists, otherwise generate synthetic Kerala data
    if os.path.exists('insurance_claims.csv'):
        raw = pd.read_csv('insurance_claims.csv')
        df = pd.DataFrame()
        df['Age'] = raw['age']
        df['Claim_Amount'] = raw['total_claim_amount']
        df['Coverage_Limit'] = raw['umbrella_limit']
        df['Policy_Type'] = np.random.choice(['Motor', 'Health', 'Agriculture', 'Life'], size=len(raw))
        df['Days_Since_Purchase'] = raw['months_as_customer'] * 30
        unique_cities = raw['incident_city'].unique()
        city_map = {city: kerala_districts[i % len(kerala_districts)] for i, city in enumerate(unique_cities)}
        df['Region'] = raw['incident_city'].map(city_map)
        df['Fraud_Reported'] = raw['fraud_reported'].map({'Y': 1, 'N': 0})
    else:
        n = 2000
        df = pd.DataFrame({
            'Age': np.random.randint(18, 75, n),
            'Claim_Amount': np.random.uniform(5000, 1500000, n),
            'Coverage_Limit': np.random.uniform(50000, 2000000, n),
            'Policy_Type': np.random.choice(['Motor', 'Health', 'Agriculture', 'Life'], n),
            'Days_Since_Purchase': np.random.randint(1, 1000, n),
            'Region': np.random.choice(kerala_districts, n)
        })
        df['Fraud_Reported'] = 0
        df.loc[(df['Claim_Amount'] > df['Coverage_Limit'] * 0.7) & (df['Days_Since_Purchase'] < 120), 'Fraud_Reported'] = 1

    ct = ColumnTransformer([('num', StandardScaler(), ['Age', 'Claim_Amount', 'Days_Since_Purchase', 'Coverage_Limit']), ('cat', OneHotEncoder(handle_unknown='ignore'), ['Policy_Type', 'Region'])])
    X = df[['Age', 'Claim_Amount', 'Policy_Type', 'Days_Since_Purchase', 'Coverage_Limit', 'Region']]
    y = df['Fraud_Reported']
    X_processed = ct.fit_transform(X)
    if HAS_SMOTE: X_res, y_res = SMOTE().fit_resample(X_processed, y)
    else: X_res, y_res = X_processed, y
    model = xgb.XGBClassifier(eval_metric='logloss', n_estimators=150).fit(X_res, y_res)
    return model, ct, kerala_districts

MODEL, PREPROCESSOR, KERALA_REGIONS = train_enterprise_model()

# --- 4. UI TEMPLATES ---
HEADER = """
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
<script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap" rel="stylesheet">
<style>
    body { font-family: 'Plus Jakarta Sans', sans-serif; background: #010409; color: #f0f6fc; }
    .glass { background: rgba(22, 27, 34, 0.7); backdrop-filter: blur(15px); border: 1px solid #30363d; }
    #map { height: 350px; border-radius: 1.5rem; border: 1px solid #30363d; }
    input, select { background: #0d1117 !important; border: 1px solid #30363d !important; color: white !important; }
</style>
"""

DASHBOARD_UI = f"<!DOCTYPE html><html><head>{HEADER}</head><body>" + """
<div class="flex h-screen overflow-hidden">
    <aside class="w-72 glass border-r border-slate-800 flex flex-col p-8 text-center">
        <h2 class="text-2xl font-black text-indigo-500 mb-8 italic">SHIELD PRO</h2>
        <div class="bg-indigo-600/10 p-5 rounded-2xl border border-indigo-500/20 mb-8">
            <h4 class="font-bold text-sm">{{ user.full_name }}</h4>
            <span class="text-[9px] px-2 py-0.5 rounded bg-indigo-500 text-white uppercase font-black">{{ user.role }}</span>
        </div>
        <a href="/logout" class="bg-red-500/10 text-red-500 p-4 rounded-xl text-center text-xs font-bold mt-auto hover:bg-red-500 hover:text-white transition-all">Logout Session</a>
    </aside>

    <main class="flex-1 p-10 overflow-y-auto">
        <div class="grid grid-cols-12 gap-8">
            <div class="col-span-8 space-y-6">
                <div class="glass p-4 rounded-3xl"><div id="map"></div></div>
                <div class="glass rounded-3xl p-8 overflow-x-auto">
                    <h3 class="font-bold mb-4 uppercase text-xs tracking-widest text-slate-500">Kerala Audit Command Ledger</h3>
                    <table class="w-full text-[11px] text-left">
                        <thead class="text-slate-500 border-b border-white/5 uppercase">
                            <tr><th>Claimant</th><th>District</th><th>Claim</th><th>Risk</th><th>PDF Report</th></tr>
                        </thead>
                        <tbody class="divide-y divide-white/5">
                            {% for entry in history %}
                            <tr>
                                <td class="py-4"><b>{{ entry.claimant_name }}</b> ({{ entry.claimant_age }})</td>
                                <td class="text-indigo-400 font-bold underline cursor-pointer" onclick="map.flyTo(keralaCoords['{{entry.region}}'], 11)">{{ entry.region }}</td>
                                <td>₹{{ "{:,.0f}".format(entry.amount) }}</td>
                                <td class="{{ 'text-red-400' if entry.score > 70 else 'text-green-400' }} font-black text-sm">{{ entry.score }}%</td>
                                <td><a href="/report/{{entry.id}}" class="bg-indigo-500/20 px-3 py-1 rounded text-indigo-400 font-bold">Download</a></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="col-span-4 glass rounded-3xl p-8 h-fit sticky top-0">
                <h3 class="font-bold mb-6 text-indigo-400 text-xs uppercase tracking-widest text-center">Execute New Audit</h3>
                <div class="space-y-4">
                    <input type="text" id="CName" placeholder="Claimant Name" class="w-full rounded-xl p-3 outline-none">
                    <input type="number" id="CAge" placeholder="Claimant Age" class="w-full rounded-xl p-3 outline-none">
                    <input type="number" id="CovLimit" placeholder="Coverage Limit (₹)" class="w-full rounded-xl p-3 outline-none">
                    <input type="number" id="Amount" placeholder="Claim Amount (₹)" class="w-full rounded-xl p-3 outline-none">
                    <input type="number" id="Tenure" placeholder="Tenure (Months)" class="w-full rounded-xl p-3 outline-none">
                    <select id="RegionSelect" class="w-full rounded-xl p-3 outline-none">
                        {% for r in kerala_regions %}<option value="{{r}}">{{r}}</option>{% endfor %}
                    </select>
                    <button onclick="runAudit()" id="auditBtn" class="w-full bg-indigo-600 py-4 rounded-xl font-black shadow-lg shadow-indigo-600/20 transition-all uppercase text-xs">Run Neural Scan</button>
                </div>
            </div>
        </div>
    </main>
</div>
<script>
    const keralaCoords = { 'Thiruvananthapuram': [8.5241, 76.9366], 'Kollam': [8.8932, 76.6141], 'Pathanamthitta': [9.2648, 76.7870], 'Alappuzha': [9.4981, 76.3388], 'Kottayam': [9.5916, 76.5221], 'Idukki': [9.9189, 77.1025], 'Ernakulam': [9.9312, 76.2673], 'Thrissur': [10.5276, 76.2144], 'Palakkad': [10.7867, 76.6547], 'Malappuram': [11.0510, 76.0711], 'Kozhikode': [11.2588, 75.7804], 'Wayanad': [11.6854, 76.1320], 'Kannur': [11.8745, 75.3704], 'Kasaragod': [12.4996, 74.9869] };
    let map = L.map('map').setView([10.5, 76.5], 7);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);

    let heatData = [{% for e in history %} [...keralaCoords['{{e.region}}'], {{e.score / 100}}], {% endfor %}];
    L.heatLayer(heatData, {radius: 25, blur: 15}).addTo(map);

    async function runAudit() {
        const payload = { claimant_name: document.getElementById('CName').value, claimant_age: parseInt(document.getElementById('CAge').value), Claim_Amount: parseFloat(document.getElementById('Amount').value), Coverage_Limit: parseFloat(document.getElementById('CovLimit').value), Tenure_Months: parseInt(document.getElementById('Tenure').value), Region: document.getElementById('RegionSelect').value };
        if(!payload.claimant_name || !payload.claimant_age) return Swal.fire('Error', 'Fill all fields', 'error');
        
        const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
        const result = await res.json();
        
        Swal.fire({ title: 'Risk Calculated', text: `Score: ${result.score}% for ${payload.claimant_name}`, icon: result.score > 70 ? 'warning' : 'success' }).then(() => location.reload());
    }
</script>
</body></html>
"""

# --- 5. SERVER ROUTES ---

@app.route('/')
def index():
    return render_template_string("""<!DOCTYPE html><html><head>""" + HEADER + """</head><body><div class="min-h-screen flex items-center justify-center p-6 bg-slate-950"><div class="glass max-w-lg w-full rounded-3xl p-10 shadow-2xl text-center">
        {% with messages = get_flashed_messages() %}{% if messages %}{% for m in messages %}<div class="text-red-500 text-xs mb-4">{{m}}</div>{% endfor %}{% endif %}{% endwith %}
        <h1 class="text-4xl font-black mb-10 italic">SHIELD<span class="text-indigo-500">PRO</span></h1><form action="/auth" method="POST" class="space-y-4"><input type="hidden" name="action" id="action" value="login"><div id="reg-fields" class="hidden space-y-4"><input name="full_name" placeholder="Full Name" class="w-full rounded-xl p-3 outline-none"><input name="email" type="email" placeholder="Email" class="w-full rounded-xl p-3 outline-none"><input name="phone" placeholder="Phone (+91...)" class="w-full rounded-xl p-3 outline-none"></div><input name="username" placeholder="Username" required class="w-full rounded-xl p-3 outline-none"><input name="password" type="password" placeholder="Password" required class="w-full rounded-xl p-3 outline-none"><button type="submit" class="w-full bg-indigo-600 py-4 rounded-xl font-black">Authorize</button><p class="text-xs text-slate-500 mt-4 cursor-pointer" onclick="document.getElementById('reg-fields').classList.toggle('hidden'); document.getElementById('action').value = document.getElementById('action').value == 'login' ? 'register' : 'login';">New User? Register Profile</p></form></div></div></body></html>""")

@app.route('/auth', methods=['POST'])
def auth():
    action = request.form.get('action')
    u = request.form.get('username')
    p = request.form.get('password')
    
    if action == 'register':
        email = request.form.get('email')
        # CHECK FOR DUPLICATES
        existing = User.query.filter((User.username == u) | (User.email == email)).first()
        if existing:
            flash("Username or Email already exists.")
            return redirect(url_for('index'))
            
        try:
            hashed = bcrypt.generate_password_hash(p).decode('utf-8')
            db.session.add(User(username=u, password=hashed, full_name=request.form.get('full_name'), email=email, phone=request.form.get('phone')))
            db.session.commit()
            flash("Account Created. Please Login.")
            return redirect(url_for('index'))
        except IntegrityError:
            db.session.rollback()
            flash("Database Error.")
            return redirect(url_for('index'))

    user = User.query.filter_by(username=u).first()
    if user and bcrypt.check_password_hash(user.password, p):
        session['user_id'] = user.id
        return redirect(url_for('dashboard'))
    flash("Invalid Login.")
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect(url_for('index'))
    user = User.query.get(session['user_id'])
    history = AuditLedger.query.order_by(AuditLedger.timestamp.desc()).all()
    return render_template_string(DASHBOARD_UI, user=user, history=history, kerala_regions=KERALA_REGIONS)

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    data = request.json
    input_df = pd.DataFrame([{ 'Age': data['claimant_age'], 'Claim_Amount': data['Claim_Amount'], 'Policy_Type': 'Motor', 'Days_Since_Purchase': data['Tenure_Months'] * 30, 'Coverage_Limit': data['Coverage_Limit'], 'Region': data['Region'] }])
    
    transformed_data = PREPROCESSOR.transform(input_df)
    score = round(float(MODEL.predict_proba(transformed_data)[:, 1][0]) * 100, 1)
    
    # Secure Hash for Audit Integrity
    msg = f"{data['claimant_name']}-{data['Region']}-{score}"
    safety = hmac.new(INTEGRITY_SECRET, msg.encode(), hashlib.sha256).hexdigest()
    
    db.session.add(AuditLedger(claimant_name=data['claimant_name'], claimant_age=data['claimant_age'], region=data['Region'], amount=data['Claim_Amount'], coverage_limit=data['Coverage_Limit'], tenure_months=data['Tenure_Months'], score=score, integrity_hash=safety, auditor_id=session['user_id']))
    db.session.commit()
    return jsonify({'score': score})

@app.route('/report/<int:id>')
def generate_report(id):
    audit = AuditLedger.query.get(id)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "SHIELD PRO AUDIT CERTIFICATE", ln=True, align='C')
    pdf.set_font("Helvetica", "", 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Claimant: {audit.claimant_name} | Age: {audit.claimant_age} | District: {audit.region}", ln=True)
    pdf.cell(0, 10, f"Claim: INR {audit.amount:,.2f} | Registered Limit: INR {audit.coverage_limit:,.2f}", ln=True)
    pdf.set_font("Helvetica", "B", 14)
    if audit.score > 70:
        pdf.set_text_color(255, 0, 0)
    else:
        pdf.set_text_color(0, 128, 0)
    pdf.cell(0, 15, f"FRAUD PROBABILITY: {audit.score}%", ln=True)
    
    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return send_file(out, as_attachment=True, download_name=f"Audit_{audit.id}.pdf", mimetype="application/pdf")

@app.route('/logout')
def logout(): 
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context(): 
        db.create_all()
    app.run(debug=True)