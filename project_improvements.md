# DriftGuard MLOps Pipeline — Improvement Roadmap
> A component-by-component audit of what can be fixed, upgraded, or used more effectively to make this a genuinely impressive, resume-grade project.

---

## Executive Summary

The project has a **solid, well-designed architecture**. All the key tools are present — Kafka, MLflow, Optuna, Evidently, Pandera, XGBoost, FastAPI. The core loop (stream → validate → detect drift → retrain → promote) works end-to-end.

The main issues are **efficiency gaps** (tools present but underused), **code-level redundancies**, **performance bottlenecks** (schema and drift checks running far more than intended), and **missing observability/robustness features** that would elevate it from "it runs" to "production-grade."

---

## 1. Evidently AI — Currently Passive, Should Be Active

**Current state:** Evidently only generates HTML files after SciPy already made the drift decision. It's a passive reporter.

**Problem:** The same KS-test is being run **twice** — once in `utils/drift_utils.py` via SciPy, and then again internally by Evidently's `DataDriftPreset`. This is redundant computation.

### What to fix

**Replace the manual SciPy loop with Evidently as the single source of truth:**

```python
# Instead of this (current):
for feature in monitored_features:
    stat, p_val = calculate_ks_test(ref_df[feature], cur_df[feature])  # manual KS
    ...
# Then ALSO running Evidently which repeats the same KS test internally

# Do this:
from evidently.metrics import DataDriftTable
from evidently import Report

report = Report(metrics=[DataDriftTable()])
result = report.run(reference_data=ref_df, current_data=cur_df)
summary = result.as_dict()  # structured result, no manual loop needed

drifted_features = [
    col for col, info in summary["metrics"][0]["result"]["drift_by_columns"].items()
    if info["drift_detected"]
]
drift_detected = len(drifted_features) >= min_drifted
```

**Add Evidently Test Suites as a pipeline gate** (currently not used at all):
```python
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns, TestColumnDrift

suite = TestSuite(tests=[
    TestShareOfDriftedColumns(lt=0.3),          # fail if >30% of cols drift
    TestColumnDrift("LIMIT_BAL", threshold=0.05),
])
suite.run(reference_data=ref_df, current_data=cur_df)
# Returns Pass/Fail — cleaner for automated pipelines
```

**Why this matters for resume:** "Used Evidently AI as the primary drift detection engine with structured programmatic output" is stronger than "generated HTML reports."

---

## 2. Kafka — Bare Minimum Setup

**Current state:** Single broker, no UI, no monitoring, topics have 1 partition each.

### What to fix

**Add Kafka UI to `docker-compose.yml`** (8 lines, zero code changes):
```yaml
kafka-ui:
  image: provectuslabs/kafka-ui:latest
  container_name: kafka-ui
  ports:
    - "8080:8080"
  depends_on:
    - kafka
  environment:
    KAFKA_CLUSTERS_0_NAME: local
    KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:29092
```
Live dashboard at `http://localhost:8080` showing topic throughput, consumer lag, message payloads.

**Add Dead-Letter Queue consumer** — currently the DLQ is a write-only black hole. There is no component that reads from `dead-letter-topic`. A simple DLQ reader that logs rejections to MLflow would complete the loop.

**Add delivery callbacks to producer calls** in `validator.py`:
```python
# Current (fire and forget):
producer.produce(valid_topic, value=...)

# Better (confirms delivery):
def delivery_report(err, msg):
    if err:
        print(f"[Validator] Delivery FAILED: {err}")

producer.produce(valid_topic, value=..., callback=delivery_report)
```

**Why this matters for resume:** "Implemented message delivery guarantees and consumer lag monitoring" vs "used Kafka."

---

## 3. MLflow — Underutilized Model Registry

**Current state:** MLflow is used for experiment tracking and model staging. The `stage` system (Production/Archived) is used correctly. However, model metadata is sparse and the deprecated stage API is being used.

### What to fix

**The staging API used is deprecated in MLflow 3.x.** `transition_model_version_stage()` is deprecated. The new approach uses Model Aliases:
```python
# Current (deprecated):
client.transition_model_version_stage(name=model_name, version=v, stage="Production")

# Modern MLflow 3.x approach:
client.set_registered_model_alias(model_name, "champion", version=v)
# Load with: mlflow.xgboost.load_model(f"models:/{model_name}@champion")
```

**Log the confusion matrix as an artifact** — `train_baseline.py` computes `Confusion_Matrix` but only prints it. Log it to MLflow:
```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax)
fig.savefig("reports/confusion_matrix.png")
mlflow.log_artifact("reports/confusion_matrix.png")
```

**Log model input signature** for schema enforcement:
```python
from mlflow.models.signature import infer_signature
signature = infer_signature(X_train, model.predict(X_train))
mlflow.xgboost.log_model(model, "model", signature=signature, ...)
```

**Why this matters for resume:** Using deprecated APIs signals unawareness of the tool's current state. Model aliases and signatures are production-grade MLflow practices.

---

## 4. Pandera Schema — Incomplete Coverage

**Current state:** The schema in `utils/schema.py` only validates **12 out of 23** features that the model actually uses. `PAY_4`, `PAY_5`, `PAY_6`, `BILL_AMT3`–`BILL_AMT6`, `PAY_AMT3`–`PAY_AMT6` have no validation rules.

**Problem:** Invalid values in unmonitored columns pass through the validator silently and enter `valid-data-topic`, potentially corrupting model training data.

### What to fix

Add the missing columns to `credit_schema`:
```python
"PAY_4": pa.Column(float),
"PAY_5": pa.Column(float),
"PAY_6": pa.Column(float),
"BILL_AMT3": pa.Column(float),
"BILL_AMT4": pa.Column(float),
"BILL_AMT5": pa.Column(float),
"BILL_AMT6": pa.Column(float),
"PAY_AMT3": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
"PAY_AMT4": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
"PAY_AMT5": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
"PAY_AMT6": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0)),
```

**Log DLQ rejection counts to MLflow** — currently rejections are just printed. Tracking the rejection rate over time is valuable observability.

**Why this matters for resume:** Incomplete schema validation is a real production bug. Fixing it demonstrates attention to data quality at the system level, not just surface level.

---

## 5. Retraining Engine — Retrains on Too Little Data

**Current state:** The retraining engine trains exclusively on the `data_snapshot` from the Kafka event payload — which is only the **current window of 500 rows**. This is a tiny dataset for XGBoost with Optuna HPO.

**Problem:** 500 rows → train/val/holdout splits → ~300 rows for training. Running 30 Optuna trials on 300 rows is computationally wasteful and statistically unreliable.

### What to fix

**Accumulate all valid records** in the drift monitor's buffer and send the full historical dataset in the retrain event, not just the current sliding window. Alternatively, persist valid records to disk/parquet and load them in the retraining engine.

**Add a minimum data size guard:**
```python
if len(snapshot_data) < 1000:
    print(f"[Retrainer] Too few samples ({len(snapshot_data)}). Skipping retraining.")
    continue
```

**Reduce Optuna trials dynamically** based on data size — 30 trials on 300 rows is overkill and slow.

**Why this matters for resume:** Training a production model on 300 rows is a conceptual error. Fixing it shows understanding of the sample size requirements for HPO.

---

## 6. FastAPI — No Prediction Logging

**Current state:** The API serves predictions but logs nothing. Every prediction is a black box — you have no record of what was predicted, when, or for what input.

**Problem:** Without prediction logging, you cannot do prediction drift monitoring, audit trails, or A/B testing comparisons.

### What to fix

**Log predictions to a file or database:**
```python
import json
from datetime import datetime

@app.post("/predict")
def predict(record: CreditRecord):
    ...
    pred = int(cached_model.predict(df)[0])
    prob = float(cached_model.predict_proba(df)[0][1])

    # Log prediction
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": record.dict(),
        "prediction": pred,
        "probability": round(prob, 4)
    }
    with open("data/prediction_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": pred, "probability": round(prob, 4)}
```

**Add `/metrics` endpoint** returning model version, total predictions, prediction distribution — useful for dashboarding and demonstrates production thinking.

**Add model version info to prediction response:**
```python
return {
    "prediction": pred,
    "probability": round(prob, 4),
    "model_version": current_model_version  # track which version made the prediction
}
```

**Why this matters for resume:** "Implemented prediction audit logging and served model version metadata" is a tangible production feature that most academic projects skip.

---

## 7. Producer — Mostly Complete, One Missing Feature

**Current state:** `producer.py` already has:
- ✅ `delivery_report` callback on every `producer.produce()` call
- ✅ Configurable delay via `producer_delay_ms` in `config.yaml`
- ✅ Progress logging every 500 rows
- ✅ `producer.flush()` in finally block

**What is still missing:**
- ❌ **No `--corrupt-rate` injection flag** for intentionally sending malformed records to test the DLQ path end-to-end. This would let you demo the validator's DLQ routing on demand.
- ❌ **No total produced / failed count at the end** — the delivery callback prints failures as they happen but doesn't summarize them.

**Why this matters for resume:** The producer is already solid. Adding a controlled corruption injection flag is the one addition that would make the DLQ path fully demonstrable in a showcase.

---

## 8. Config & Architecture — Minor Issues

### Hardcoded values in `train_baseline.py`
The experiment name, model name, and tracking URI are **hardcoded at module level**, not loaded from `config.yaml`. Every other component uses `load_config()`. This is inconsistent.

```python
# Current (inconsistent):
EXPERIMENT_NAME = "Drift-Guard-MLOPS-Pipeline"
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Should be:
config = load_config()
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])
```

### No `__init__.py` files
The `consumer/`, `model/`, `utils/`, `producer/` directories have no `__init__.py`. This works because of how Python's `-m` flag handles modules, but adding them makes imports cleaner and the package structure explicit.

### `requirements.txt` has unused packages
`Flask`, `litestar`, `litestar-htmx`, `huey`, `Faker`, `skops`, `waitress` are installed but not used anywhere in the codebase. This inflates the environment and creates potential security surface.

---

## 9. What's Missing Entirely

These are features that would significantly distinguish the project:

| Missing Feature | Why It Matters |
|---|---|
| **Tests** (`pytest`) | Zero test coverage. Even 3-4 unit tests for `validate_record()`, `calculate_ks_test()`, and `evaluate_and_promote()` demonstrates engineering discipline |
| **DLQ Consumer** | `dead-letter-topic` is written to but never read. A simple consumer that logs/alerts on DLQ spikes completes the architecture |
| **Prediction drift monitoring** | Use Evidently on `prediction_log.jsonl` to detect when the model's output distribution shifts — a separate concern from input data drift |
| **GitHub Actions CI** | A workflow that runs `pytest` and validates the schema on PR would make this a complete MLOps project, not just an ML project |
| **`docker-compose` for the full app** | Currently only Kafka/Zookeeper are dockerized. Adding services for the consumers and API would make the whole thing one-command deployable |
| **SHAP explainability** | Log SHAP feature importance plots to MLflow during retraining — demonstrates model interpretability awareness |

---

## 10. Performance — Schema & Drift Detection Running Too Frequently

This is the most impactful runtime optimization in the entire pipeline. Both the validator and the drift monitor are doing significantly more work than necessary on every message.

---

### Problem A — Pandera Validates One Row at a Time (Validator)

**Current behavior** in [`consumer/validator.py`](consumer/validator.py):

```python
# This runs for EVERY single Kafka message:
is_valid, err_msg = validate_record(record)  # ← called 10,000+ times
```

And inside `utils/schema.py`:
```python
def validate_record(record: dict):
    df = pd.DataFrame([record])      # ← creates a new 1-row DataFrame per message
    credit_schema.validate(df)       # ← full Pandera validation per message
```

**The cost:** Creating a `pd.DataFrame` from a dict is not free. With 15,000 streaming records:
- 15,000 DataFrame constructions
- 15,000 full Pandera schema validations
- Pandera internally does dtype checks, constraint checks, and coercion on every call

**Pandera is a batch-first library** — it is designed to validate a DataFrame of N rows, not to be called N times with 1 row each.

### Fix — Micro-batch validation

Accumulate records into a buffer in the validator, validate as a batch, then route the whole batch:

```python
MICRO_BATCH_SIZE = 50  # tune based on latency tolerance

batch_buffer = []

while True:
    msg = consumer.poll(timeout=1.0)
    if msg is None:
        # Flush partial batch on idle
        if batch_buffer:
            _process_batch(batch_buffer, producer, valid_topic, dlq_topic)
            batch_buffer = []
        continue

    record = json.loads(msg.value().decode("utf-8"))
    batch_buffer.append(record)

    if len(batch_buffer) >= MICRO_BATCH_SIZE:
        _process_batch(batch_buffer, producer, valid_topic, dlq_topic)
        batch_buffer = []

def _process_batch(records, producer, valid_topic, dlq_topic):
    df = pd.DataFrame(records)           # ← 1 DataFrame for 50 rows
    try:
        credit_schema.validate(df, lazy=True)  # lazy=True collects all errors at once
        for record in records:
            producer.produce(valid_topic, value=json.dumps(record).encode())
    except pa.errors.SchemaErrors as exc:
        # exc.failure_cases tells you exactly which row indices failed
        failed_indices = set(exc.failure_cases["index"].dropna().astype(int).tolist())
        for i, record in enumerate(records):
            if i in failed_indices:
                producer.produce(dlq_topic, value=json.dumps({...}).encode())
            else:
                producer.produce(valid_topic, value=json.dumps(record).encode())
```

**Result:** 15,000 records → **300 Pandera calls** (50x reduction). Each call validates 50 rows at once, which is the pattern Pandera is optimized for. The `lazy=True` flag is critical — it collects all row-level failures instead of throwing on the first one, giving you per-row error attribution within the batch.

**Latency tradeoff:** A micro-batch of 50 means a record waits at most until 49 others arrive or the idle flush triggers. At the `producer_delay_ms: 100` rate, this is a ~5 second max wait — acceptable for a batch analytics pipeline.

---

### Problem B — Detection Interval Logic Is Broken (Drift Monitor)

**Current behavior** in [`consumer/drift_monitor.py`](consumer/drift_monitor.py):

```python
current_buffer = deque(maxlen=cur_size)  # maxlen=500
records_since_last_check = 0

# Inside the message loop:
if ref_df is None:
    ...  # filling reference window
else:
    current_buffer.append(record)        # deque auto-drops oldest when full
    if len(current_buffer) == cur_size:  # True only when buffer is exactly 500
        records_since_last_check += 1    # ← increments on EVERY record after warm-up

        if records_since_last_check >= interval:  # interval = 100
```

**The bug:** Once the `current_buffer` is full (500 records), `len(current_buffer) == 500` is **true for every subsequent record** (because the deque stays at maxlen). So `records_since_last_check` increments on every single message. The check triggers at message 100, resets to 0, then triggers again at message 200, 300...

**Actual behavior:** After warm-up, drift detection runs **every 100 messages** — not "every interval records" as intended. With 10,000 streamed records after warm-up, drift detection runs **100 times**, each triggering two Evidently reports, two MLflow runs, and potentially a retrain event.

**Intended behavior (from config comment):** `# run KS-test every N rows, not every row` — clearly the intent was to run it much less frequently.

### Fix — Track total records seen, not buffer fill events

```python
total_records_seen = 0   # replaces records_since_last_check

# Inside the message loop (after reference window is full):
current_buffer.append(record)
total_records_seen += 1  # always increment

# Only run drift check when buffer is full AND interval has passed
if len(current_buffer) >= cur_size and total_records_seen % interval == 0:
    cycle += 1
    # ... run drift detection
```

**Result:** With `interval=100` and 10,000 records, drift detection runs **100 times** (same), but the fix makes it predictable and correct. To run it less frequently, just increase `detection_interval` in `config.yaml` to `500` — then it runs only 20 times over 10,000 records, each with a fresh 500-record window.

**Recommended `config.yaml` tuning:**
```yaml
drift:
  detection_interval: 500   # was 100 — check every 500 new valid records
```

---

### Problem C — Double KS-Test Computation (Already in Section 1)

As noted in Section 1, SciPy runs KS-test manually, then Evidently's `DataDriftPreset` runs the same KS-test internally. Combined with Problem B above, if drift detection fires 100 times:
- SciPy KS-test: **400 calls** (4 features × 100 cycles)
- Evidently KS-test: **400 calls** (same features, same cycles, internally)
- **Total: 800 KS-test computations** where 400 were needed

**Combined fix:** Replacing SciPy with Evidently (Section 1) + increasing `detection_interval` to 500 reduces this to **~80 KS computations** — a **10x reduction**.

---

### Performance Summary

| Component | Current calls (15k records) | After optimization | Reduction |
|---|---|---|---|
| Pandera `validate()` | 15,000 | 300 | **50×** |
| Drift detection cycles | ~100 | ~20 | **5×** |
| KS-test computations | ~800 | ~80 | **10×** |
| Evidently `Report.run()` | ~200 | ~20 | **10×** |

---

## 11. Retraining Trigger & Challenger MLflow Logging — Audit

### What triggers retraining

The full chain is:

```
drift_monitor.py
  └─ KS-test p-value < 0.05 on >= 2 features
        └─ drift_detected = True
              └─ publishes to retrain-events-topic
                    └─ retraining_engine.py consumes event
                          └─ runs Optuna HPO → registers challenger
                                └─ calls promoter.py → champion vs challenger on holdout
```

The trigger payload sent on the Kafka topic contains:
```json
{
  "cycle": 1,
  "timestamp": 1234567890.0,
  "drifted_features": ["LIMIT_BAL", "AGE"],
  "data_snapshot": [ ...500 records... ]
}
```

---

### Problem 1 — No Guard Against Duplicate / Cascading Retrain Events

**Current behavior:** If drift is detected on cycle 3, 4, and 5 consecutively (which is likely since the current_buffer is a sliding window — the same drifted data stays in it), three separate retrain events are published to `retrain-events-topic`. The retraining engine will process all three sequentially, launching **three full Optuna HPO runs** (30 trials × 3 = 90 trials) on nearly identical data.

**There is no cooldown, lock, or deduplication mechanism.**

### Fix — Add a retraining cooldown flag

```python
# In drift_monitor.py, add after cycle counter:
retrain_cooldown_cycles = 0
COOLDOWN = 3  # don't retrigger for N cycles after a retrain event

# In the drift check block:
if drift_detected and retrain_cooldown_cycles == 0:
    producer.produce(retrain_topic, ...)
    retrain_cooldown_cycles = COOLDOWN
elif retrain_cooldown_cycles > 0:
    retrain_cooldown_cycles -= 1
    print(f"[DriftMonitor] Drift detected but in cooldown ({retrain_cooldown_cycles} cycles remaining).")
```

---

### Problem 2 — Challenger Model Has No Holdout Evaluation Metrics in MLflow

This is the most significant MLflow logging gap. After a full retraining cycle, here is what is and isn't logged:

**What IS logged for the challenger:**

| Location | What's logged |
|---|---|
| `retraining_engine.py` parent run | `snapshot_size` (param), `cycle` (tag), `drifted_features` (tag) |
| `train_optuna.py` — each trial nested run | `f1`, `accuracy`, `precision`, `recall` on **validation split** |
| `train_optuna.py` — parent run | `best_f1` (best Optuna trial val score), `best_*` params |
| `promoter.py` → `registry.py` | `metric_gap` (float), `promotion_decision` (tag) |

**What is NOT logged (the gaps):**

| Missing Metric | Where it's computed | What happens to it |
|---|---|---|
| Challenger **holdout F1** | `promoter.py` line 37: `challenger_f1 = f1_score(...)` | Only printed to console, never logged |
| Champion **holdout F1** | `promoter.py` line 51: `champion_f1 = f1_score(...)` | Only printed to console, never logged |
| Final model **val metrics** | `train_optuna.py` trains final model on `X_full_train` (train+val combined) | **Final model has zero evaluation** — it's registered with only HPO trial metrics |
| `ROC-AUC` | Not computed anywhere in the retraining path | Entirely missing |
| Training data shape | `X_train.shape`, `X_val.shape` | Not logged |

### The Critical Disconnect

The `best_f1` logged in MLflow (from Optuna trial on val split) is **not the same model** as the registered challenger. The final model is retrained on `X_full_train = concat(X_train, X_val)` — a different dataset. The logged F1 belongs to an intermediate Optuna trial model, not the registered artifact.

```python
# train_optuna.py — what gets logged:
mlflow.log_metric("best_f1", study.best_value)  # ← from HPO trial on val split

# Then the FINAL model is trained here — with NO metrics logged:
final_model = XGBClassifier(**best_params)
final_model.fit(X_full_train, y_full_train)  # ← train+val combined, never evaluated
return final_model, best_params               # ← registered in MLflow without metrics
```

### Fix — Log holdout metrics for challenger in promoter.py

```python
# In promoter.py, after challenger_f1 is computed:
from sklearn.metrics import roc_auc_score

challenger_prob = challenger_model.predict_proba(X_holdout)[:, 1]
challenger_roc = roc_auc_score(y_holdout, challenger_prob)

# Log to the challenger's MLflow run:
client.log_metric(challenger_run_id, "holdout_f1", challenger_f1)
client.log_metric(challenger_run_id, "holdout_roc_auc", challenger_roc)

# If champion exists, also log champion's holdout metrics:
if champion_version_obj:
    champion_prob = champion_model.predict_proba(X_holdout)[:, 1]
    champion_roc = roc_auc_score(y_holdout, champion_prob)
    client.log_metric(challenger_run_id, "champion_holdout_f1", champion_f1)
    client.log_metric(challenger_run_id, "champion_holdout_roc_auc", champion_roc)
    client.log_metric(challenger_run_id, "metric_gap_f1", metric_gap)
```

### Fix — Evaluate final model before registering in train_optuna.py

```python
# After training final_model on X_full_train:
# Re-evaluate on val set to get honest final model metrics
y_pred_final = final_model.predict(X_val)
final_f1 = f1_score(y_val, y_pred_final)
final_acc = accuracy_score(y_val, y_pred_final)

mlflow.log_metrics({
    "final_model_f1": final_f1,
    "final_model_accuracy": final_acc,
    "n_optuna_trials": n_trials,
    "training_rows": len(X_full_train),
})
```

---

### Problem 3 — `X_eval` / `y_eval` Created But Never Used (Dead Variable)

**Current code in `retraining_engine.py` lines 62–68:**

```python
# 2. Perform train/validation/holdout splits
X_train_val, X_eval, y_train_val, y_eval = train_test_split(
    X_snap, y_snap, test_size=holdout_split, random_state=42, stratify=y_snap
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_split, random_state=42, stratify=y_train_val
)
# Then:
final_model, best_params = run_hpo(X_train, y_train, X_val, y_val, ...)  # X_eval never passed
```

**`X_eval` and `y_eval` are split off (consuming 15% of the already-tiny 500-row snapshot) and then never passed anywhere.** They are created and immediately garbage collected. The retraining engine splits off a holdout set for the retraining cycle but never evaluates the final model against it.

**Impact:** The retraining cycle trains on only ~357 rows (70% of 85% of 500) instead of the ~425 rows it could use if the holdout were evaluated post-training and incorporated. With 500 rows this matters significantly.

**Fix:** Either pass `X_eval` to `run_hpo` for final model evaluation, or use it after HPO to independently evaluate the registered challenger:

```python
final_model, best_params = run_hpo(X_train, y_train, X_val, y_val, n_trials=n_trials)

# Evaluate final model on the held-out eval set:
y_eval_pred = final_model.predict(X_eval)
eval_f1 = f1_score(y_eval, y_eval_pred)
mlflow.log_metric("retraining_eval_f1", eval_f1)  # now X_eval is actually used
```

---

### Problem 4 — Stratified Split Will Crash on Small or Imbalanced Windows

The `stratify=y_snap` on line 64 uses `y_snap` from the 500-row current window. Credit default data is imbalanced (~22% positive class). On a 500-row window, the positive class has ~110 samples. With `holdout_split=0.15` → holdout gets ~16 positive samples. With a further `val_split=0.15` → the val set may get as few as 2–3 positive samples.

**`train_test_split` with `stratify` requires at least 2 members per class per resulting split.** When the window is small or has an unusual drift-induced class distribution, this will raise a `ValueError: The least populated class in y has only N members, which is too few.` and crash the entire retraining cycle silently (the exception is not caught).

**There is no try/except around the split calls.**

**Fix:**
```python
try:
    X_train_val, X_eval, y_train_val, y_eval = train_test_split(
        X_snap, y_snap, test_size=holdout_split, random_state=42, stratify=y_snap
    )
except ValueError:
    # Fall back to non-stratified split if class distribution is too small
    print("[Retrainer] Warning: Stratified split failed, using random split.")
    X_train_val, X_eval, y_train_val, y_eval = train_test_split(
        X_snap, y_snap, test_size=holdout_split, random_state=42
    )
```

---

### Summary of Retraining / Logging Issues

| # | Issue | Severity | Fix effort |
|---|---|---|---|
| 1 | No retrain cooldown → cascading duplicate HPO runs | 🔴 High | Low (5 lines) |
| 2 | Challenger holdout F1 never logged to MLflow | 🔴 High | Low (4 lines) |
| 3 | Champion holdout F1 never logged | 🟡 Medium | Low (3 lines) |
| 4 | Final model (on X_full_train) has zero evaluation metrics | 🔴 High | Low (8 lines) |
| 5 | ROC-AUC never computed anywhere in the retraining path | 🟡 Medium | Low (2 lines) |
| 6 | **`X_eval`/`y_eval` created but never used (dead variable)** | 🔴 High | Low (3 lines) |
| 7 | **Stratified split crashes on small/imbalanced windows** | 🔴 High | Low (5 lines) |
| 8 | Training data shape/size not logged | 🟢 Low | Trivial |

---

## Priority Order for Resume Impact

```
HIGH IMPACT / LOW EFFORT
├── Fix deprecated MLflow staging API → aliases           (2 lines changed)
├── Add Kafka UI to docker-compose                        (8 lines added)
├── Complete Pandera schema coverage                      (11 lines added)
├── Add prediction logging to /predict                    (10 lines added)
├── Fix hardcoded values in train_baseline.py             (4 lines changed)
├── Log challenger holdout F1 + ROC-AUC in promoter.py   (6 lines added)
├── Evaluate final model before registering in optuna.py  (8 lines added)
├── Add retrain cooldown guard in drift_monitor.py        (5 lines added)
├── Fix X_eval dead variable → evaluate challenger        (3 lines changed)
├── Add stratify crash guard in retraining_engine.py      (5 lines changed)
└── Remove unused packages from requirements.txt          (cleanup)

HIGH IMPACT / MEDIUM EFFORT
├── Fix detection_interval logic bug in drift_monitor.py  (4 lines changed)
├── Switch Pandera to micro-batch validation (50×)        (refactor validator.py)
├── Replace dual KS-test with Evidently as single engine  (refactor drift_monitor.py)
├── Add Kafka delivery callbacks                          (5 lines per produce call)
├── Add DLQ consumer                                      (new file ~50 lines)
└── Add pytest unit tests                                 (new file ~80 lines)

HIGH IMPACT / HIGH EFFORT
├── Accumulate training data beyond current window        (redesign retrain payload)
├── Add prediction drift monitoring via Evidently         (new pipeline stage)
└── GitHub Actions CI pipeline                            (new .yml file)
```

---

## What This Becomes After Fixes

**Current resume bullet:**
> "Built an event-driven MLOps pipeline using Kafka, MLflow, Evidently, and XGBoost with automated retraining on drift detection."

**After fixes:**
> "Built a production-grade event-driven MLOps pipeline with Kafka-backed streaming (with cooldown-gated retrain triggers), Pandera micro-batch schema validation (50× efficiency gain), Evidently AI as the unified drift detection and reporting engine, MLflow model registry with full holdout F1/ROC-AUC logging for both champion and challenger, Optuna HPO retraining with proper final model evaluation, prediction audit logging, and a FastAPI serving layer — fully observable with Kafka UI and MLflow dashboards."

The architecture is already strong. These changes make the **implementation** match the **ambition** of the architecture.
