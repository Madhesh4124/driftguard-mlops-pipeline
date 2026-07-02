import os
import sys
# Inject project root path so uvicorn can locate top-level packages (like utils)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from utils.config_loader import load_config


# Global variable to cache the loaded model and its version
cached_model = None
cached_model_version = None
features_order = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

class CreditRecord(BaseModel):
    LIMIT_BAL: float
    SEX: float
    EDUCATION: float
    MARRIAGE: float
    AGE: float
    PAY_0: float
    PAY_2: float
    PAY_3: float
    PAY_4: float
    PAY_5: float
    PAY_6: float
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

def load_production_model():
    """
    Fetches the Champion-tagged model from MLflow Model Registry and caches it.
    """
    global cached_model, cached_model_version
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    
    model_name = config["mlflow"]["model_name"]
    model_uri = f"models:/{model_name}@champion"
    
    print(f"Attempting to load Champion model from URI: {model_uri}...")
    try:
        cached_model = mlflow.xgboost.load_model(model_uri)
        client = mlflow.tracking.MlflowClient()
        mv = client.get_model_version_by_alias(model_name, "champion")
        cached_model_version = int(mv.version)
        print(f"Champion model (v{cached_model_version}) successfully loaded and cached.")
        return True
    except Exception as e:
        print(f"Error loading model from MLflow Registry: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    success = load_production_model()
    if not success:
        print("Warning: Failed to load Production model on startup.")
    yield
    # Cleanup on shutdown (if any)
    print("Shutting down API.")

app = FastAPI(
    title="Drift-Aware Credit Default Prediction Service",
    description="Serves predictions using the Production model registered in MLflow.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
def home():
    global cached_model
    model_status_class = "status-success" if cached_model is not None else "status-error"
    model_status_text = "Production Model Loaded" if cached_model is not None else "No Model Loaded (Run baseline training)"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Drift-Guard MLOps Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-primary: #0b0f19;
                --bg-secondary: rgba(255, 255, 255, 0.03);
                --border-color: rgba(255, 255, 255, 0.06);
                --text-primary: #f3f4f6;
                --text-secondary: #9ca3af;
                --accent-cyan: #00f2fe;
                --accent-pink: #ff2a5f;
                --accent-blue: #4364f7;
                --font-sans: 'Outfit', sans-serif;
                --font-mono: 'JetBrains Mono', monospace;
            }}
            
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            
            body {{
                background-color: var(--bg-primary);
                color: var(--text-primary);
                font-family: var(--font-sans);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                overflow-x: hidden;
            }}
            
            header {{
                background: rgba(11, 15, 25, 0.8);
                backdrop-filter: blur(12px);
                border-bottom: 1px solid var(--border-color);
                padding: 1.5rem 2rem;
                position: sticky;
                top: 0;
                z-index: 10;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            header h1 {{
                font-size: 1.8rem;
                font-weight: 800;
                background: linear-gradient(135deg, var(--accent-cyan), #4facfe);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.5px;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 2rem auto;
                padding: 0 1.5rem;
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 2rem;
                width: 100%;
                flex-grow: 1;
            }}
            
            @media (max-width: 1024px) {{
                .container {{
                    grid-template-columns: 1fr;
                }}
            }}
            
            .card {{
                background: var(--bg-secondary);
                backdrop-filter: blur(10px);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }}
            
            .card-title {{
                font-size: 1.3rem;
                font-weight: 600;
                border-left: 4px solid var(--accent-cyan);
                padding-left: 0.75rem;
                letter-spacing: -0.2px;
            }}
            
            .status-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem 1rem;
                border-radius: 9999px;
                font-weight: 600;
                font-size: 0.9rem;
            }}
            
            .status-success {{
                background: rgba(0, 242, 254, 0.1);
                color: var(--accent-cyan);
                border: 1px solid rgba(0, 242, 254, 0.2);
            }}
            
            .status-error {{
                background: rgba(255, 42, 95, 0.1);
                color: var(--accent-pink);
                border: 1px solid rgba(255, 42, 95, 0.2);
            }}
            
            .btn {{
                background: linear-gradient(135deg, var(--accent-blue), #6fb1fc);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1.5rem;
                font-family: var(--font-sans);
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
                text-decoration: none;
            }}
            
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(67, 100, 247, 0.4);
            }}
            
            .btn-secondary {{
                background: transparent;
                border: 1px solid var(--border-color);
                color: var(--text-primary);
            }}
            
            .btn-secondary:hover {{
                background: rgba(255, 255, 255, 0.05);
                box-shadow: none;
            }}
            
            .form-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                gap: 1rem;
            }}
            
            .form-group {{
                display: flex;
                flex-direction: column;
                gap: 0.4rem;
            }}
            
            .form-group label {{
                font-size: 0.8rem;
                color: var(--text-secondary);
                font-weight: 600;
            }}
            
            .form-group input, .form-group select {{
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--border-color);
                border-radius: 6px;
                padding: 0.5rem;
                color: var(--text-primary);
                font-family: var(--font-mono);
                font-size: 0.9rem;
            }}
            
            .form-group select option {{
                background-color: var(--bg-primary);
                color: var(--text-primary);
            }}

            
            .form-group input:focus, .form-group select:focus {{
                border-color: var(--accent-cyan);
                outline: none;
            }}
            
            .preset-container {{
                display: flex;
                gap: 0.75rem;
                flex-wrap: wrap;
            }}
            
            .console-card {{
                background: #060913;
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 1.5rem;
                font-family: var(--font-mono);
                font-size: 0.95rem;
                color: #5af78e;
                overflow-y: auto;
                max-height: 400px;
                white-space: pre-wrap;
                flex-grow: 1;
            }}
            
            footer {{
                text-align: center;
                padding: 2rem;
                color: var(--text-secondary);
                font-size: 0.85rem;
                border-top: 1px solid var(--border-color);
            }}
            
            .highlight {{
                color: var(--accent-cyan);
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>Drift-Guard MLOps Portal</h1>
            <div>
                <span class="status-badge {model_status_class}">{model_status_text}</span>
            </div>
        </header>
        
        <div class="container">
            <!-- Left Side: System Control -->
            <div style="display: flex; flex-direction: column; gap: 2rem;">
                <div class="card">
                    <h2 class="card-title">Model Operations</h2>
                    <p style="color: var(--text-secondary); line-height: 1.5;">
                        Manage the current FastAPI serving state. Trigger an on-demand cache reload after running HPO retraining or promotion steps.
                    </p>
                    <button id="btn-reload" class="btn" onclick="reloadModel()">Reload Production Model</button>
                    <a href="/docs" target="_blank" class="btn btn-secondary">Open Swagger API Docs</a>
                    <a href="/health" target="_blank" class="btn btn-secondary">Check API Health</a>
                </div>
                
                <div class="card" style="flex-grow: 1;">
                    <h2 class="card-title">Prediction Output</h2>
                    <div id="output-console" class="console-card">// Run a prediction query to see outputs...</div>
                </div>
            </div>
            
            <!-- Right Side: Testing Playground -->
            <div class="card">
                <h2 class="card-title">Live Testing Playground</h2>
                <div class="preset-container" style="margin-bottom: 1.5rem;">
                    <span style="align-self: center; font-size: 0.9rem; color: var(--text-secondary); font-weight: 600;">Load Presets:</span>
                    <button class="btn btn-secondary" onclick="loadPreset('low')">Low-Risk Customer (Safe)</button>
                    <button class="btn btn-secondary" onclick="loadPreset('high')">High-Risk Customer (Default)</button>
                </div>
                
                <form id="prediction-form" onsubmit="submitPrediction(event)">
                    <h3 style="font-size: 1rem; margin-bottom: 1rem; color: var(--accent-cyan);">Demographics & Limit</h3>
                    <div class="form-grid" style="margin-bottom: 1.5rem;">
                        <div class="form-group">
                            <label for="LIMIT_BAL">Limit Bal</label>
                            <input type="number" id="LIMIT_BAL" step="any" required>
                        </div>
                        <div class="form-group">
                            <label for="AGE">Age</label>
                            <input type="number" id="AGE" required>
                        </div>
                        <div class="form-group">
                            <label for="SEX">Sex</label>
                            <select id="SEX">
                                <option value="1">1 (Male)</option>
                                <option value="2">2 (Female)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="EDUCATION">Education</label>
                            <select id="EDUCATION">
                                <option value="1">1 (Grad School)</option>
                                <option value="2">2 (University)</option>
                                <option value="3">3 (High School)</option>
                                <option value="4">4 (Others)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="MARRIAGE">Marriage</label>
                            <select id="MARRIAGE">
                                <option value="1">1 (Married)</option>
                                <option value="2">2 (Single)</option>
                                <option value="3">3 (Others)</option>
                            </select>
                        </div>
                    </div>
                    
                    <h3 style="font-size: 1rem; margin-bottom: 1rem; color: var(--accent-cyan);">Repayment Status (PAY_0 to PAY_6)</h3>
                    <div class="form-grid" style="margin-bottom: 1.5rem;">
                        <div class="form-group"><label for="PAY_0">PAY_0 (Sep)</label><input type="number" id="PAY_0" required></div>
                        <div class="form-group"><label for="PAY_2">PAY_2 (Aug)</label><input type="number" id="PAY_2" required></div>
                        <div class="form-group"><label for="PAY_3">PAY_3 (Jul)</label><input type="number" id="PAY_3" required></div>
                        <div class="form-group"><label for="PAY_4">PAY_4 (Jun)</label><input type="number" id="PAY_4" required></div>
                        <div class="form-group"><label for="PAY_5">PAY_5 (May)</label><input type="number" id="PAY_5" required></div>
                        <div class="form-group"><label for="PAY_6">PAY_6 (Apr)</label><input type="number" id="PAY_6" required></div>
                    </div>
                    
                    <h3 style="font-size: 1rem; margin-bottom: 1rem; color: var(--accent-cyan);">Bill Statement Amounts</h3>
                    <div class="form-grid" style="margin-bottom: 1.5rem;">
                        <div class="form-group"><label for="BILL_AMT1">Sep (1)</label><input type="number" id="BILL_AMT1" required></div>
                        <div class="form-group"><label for="BILL_AMT2">Aug (2)</label><input type="number" id="BILL_AMT2" required></div>
                        <div class="form-group"><label for="BILL_AMT3">Jul (3)</label><input type="number" id="BILL_AMT3" required></div>
                        <div class="form-group"><label for="BILL_AMT4">Jun (4)</label><input type="number" id="BILL_AMT4" required></div>
                        <div class="form-group"><label for="BILL_AMT5">May (5)</label><input type="number" id="BILL_AMT5" required></div>
                        <div class="form-group"><label for="BILL_AMT6">Apr (6)</label><input type="number" id="BILL_AMT6" required></div>
                    </div>
                    
                    <h3 style="font-size: 1rem; margin-bottom: 1rem; color: var(--accent-cyan);">Previous Paid Amounts</h3>
                    <div class="form-grid" style="margin-bottom: 2rem;">
                        <div class="form-group"><label for="PAY_AMT1">Sep (1)</label><input type="number" id="PAY_AMT1" required></div>
                        <div class="form-group"><label for="PAY_AMT2">Aug (2)</label><input type="number" id="PAY_AMT2" required></div>
                        <div class="form-group"><label for="PAY_AMT3">Jul (3)</label><input type="number" id="PAY_AMT3" required></div>
                        <div class="form-group"><label for="PAY_AMT4">Jun (4)</label><input type="number" id="PAY_AMT4" required></div>
                        <div class="form-group"><label for="PAY_AMT5">May (5)</label><input type="number" id="PAY_AMT5" required></div>
                        <div class="form-group"><label for="PAY_AMT6">Apr (6)</label><input type="number" id="PAY_AMT6" required></div>
                    </div>
                    
                    <button type="submit" class="btn" style="width: 100%;">Submit Live Prediction Request</button>
                </form>
            </div>
        </div>
        
        <footer>
            Event-Driven Drift-Aware Credit MLOps Pipeline Platform &copy; 2026
        </footer>
        
        <script>
            const presets = {{
                low: {{
                    LIMIT_BAL: 250000, AGE: 30, SEX: 2, EDUCATION: 1, MARRIAGE: 2,
                    PAY_0: 0, PAY_2: 0, PAY_3: 0, PAY_4: 0, PAY_5: 0, PAY_6: 0,
                    BILL_AMT1: 12000, BILL_AMT2: 11500, BILL_AMT3: 9800, BILL_AMT4: 10300, BILL_AMT5: 8900, BILL_AMT6: 9400,
                    PAY_AMT1: 5000, PAY_AMT2: 4000, PAY_AMT3: 3500, PAY_AMT4: 4500, PAY_AMT5: 3000, PAY_AMT6: 4000
                }},
                high: {{
                    LIMIT_BAL: 30000, AGE: 45, SEX: 1, EDUCATION: 2, MARRIAGE: 1,
                    PAY_0: 2, PAY_2: 2, PAY_3: 2, PAY_4: 2, PAY_5: 2, PAY_6: 2,
                    BILL_AMT1: 29000, BILL_AMT2: 28500, BILL_AMT3: 28000, BILL_AMT4: 27500, BILL_AMT5: 26000, BILL_AMT6: 25500,
                    PAY_AMT1: 500, PAY_AMT2: 0, PAY_AMT3: 1000, PAY_AMT4: 0, PAY_AMT5: 500, PAY_AMT6: 0
                }}
            }};
            
            function loadPreset(key) {{
                const p = presets[key];
                for (const k in p) {{
                    const el = document.getElementById(k);
                    if (el) el.value = p[k];
                }}
            }}
            
            // Load Low-Risk by default
            loadPreset('low');
            
            async function reloadModel() {{
                const consoleEl = document.getElementById('output-console');
                consoleEl.innerHTML = "Reloading model from MLflow Model Registry...";
                try {{
                    const res = await fetch('/reload', {{ method: 'POST' }});
                    const data = await res.json();
                    if (res.ok) {{
                        consoleEl.innerHTML = JSON.stringify(data, null, 2);
                        setTimeout(() => window.location.reload(), 1500);
                    }} else {{
                        consoleEl.innerHTML = "Error: " + JSON.stringify(data, null, 2);
                    }}
                }} catch (e) {{
                    consoleEl.innerHTML = "Connection Error: " + e;
                }}
            }}
            
            async function submitPrediction(e) {{
                e.preventDefault();
                const consoleEl = document.getElementById('output-console');
                consoleEl.innerHTML = "Sending prediction request...";
                
                const record = {{}};
                const fields = [
                    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
                ];
                
                fields.forEach(f => {{
                    record[f] = parseFloat(document.getElementById(f).value);
                }});
                
                try {{
                    const res = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(record)
                    }});
                    const data = await res.json();
                    if (res.ok) {{
                        const formatted = `Prediction Output:\\n` +
                                          `-----------------------------------\\n` +
                                          `Default Class Prediction: ${{data.prediction == 1 ? '⚠️ DEFAULT' : '✅ SAFE'}}\\n` +
                                          `Estimated Probability:  ${{(data.probability * 100).toFixed(2)}}%\\n\\n` +
                                          `Raw API Response:\\n` + JSON.stringify(data, null, 2);
                        consoleEl.innerHTML = formatted;
                    }} else {{
                        consoleEl.innerHTML = "Error: " + JSON.stringify(data, null, 2);
                    }}
                }} catch (e) {{
                    consoleEl.innerHTML = "Connection Error: " + e;
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/predict")
def predict(record: CreditRecord):
    global cached_model, cached_model_version
    if cached_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please try again or check the API logs."
        )
    
    # 1. Convert payload to DataFrame and align column order
    record_dict = record.dict()
    df = pd.DataFrame([record_dict])
    df = df[features_order]
    
    # 2. Make prediction
    try:
        pred = int(cached_model.predict(df)[0])
        prob = float(cached_model.predict_proba(df)[0][1])
        
        # Log prediction to file for audit and drift monitoring
        import json
        from datetime import datetime
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": record_dict,
            "prediction": pred,
            "probability": round(prob, 4),
            "model_version": cached_model_version
        }
        os.makedirs("data", exist_ok=True)
        with open("data/prediction_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        return {
            "prediction": pred,
            "probability": round(prob, 4),
            "model_version": cached_model_version
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/reload")
def reload_model():
    """
    Admin endpoint to trigger reloading the Production model.
    """
    print("Received reload request.")
    success = load_production_model()
    if success:
        return {"status": "success", "message": "Production model successfully reloaded."}
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to reload Production model from Model Registry. See logs."
        )

@app.get("/metrics")
def get_metrics():
    """
    Returns prediction serving metrics such as total predictions, prediction distribution,
    and current model version.
    """
    global cached_model_version
    import json
    log_path = "data/prediction_log.jsonl"
    
    total_predictions = 0
    distribution = {0: 0, 1: 0}
    
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    total_predictions += 1
                    pred = data.get("prediction")
                    if pred in [0, 1]:
                        distribution[pred] += 1
        except Exception as e:
            print(f"Error reading prediction logs: {e}")
            
    return {
        "model_version": cached_model_version,
        "total_predictions": total_predictions,
        "prediction_distribution": {
            "no_default": distribution[0],
            "default": distribution[1]
        }
    }

@app.get("/health")
def health_check():
    global cached_model
    return {
        "status": "healthy",
        "model_loaded": cached_model is not None
    }
