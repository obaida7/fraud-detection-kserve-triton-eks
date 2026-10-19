#!/usr/bin/env python3
"""
app.py
Real-Time Fraud Detection Operations Portal & Inference API.
Connects to Triton Inference Server / KServe v2 on EKS via ALB Ingress,
with automatic fallback to local ONNX Runtime for standalone offline demonstrations.
"""

import os
import sys
import time
import json
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import onnxruntime as ort

app = Flask(__name__, template_folder="templates")
CORS(app)

# Configuration
TRITON_URL = os.environ.get("TRITON_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "fraud_detection")
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.35"))

# Local ONNX fallback engine
LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model_repository", "fraud_detection", "1", "model.onnx"
)
LOCAL_SESSION = None
if os.path.exists(LOCAL_MODEL_PATH):
    try:
        LOCAL_SESSION = ort.InferenceSession(LOCAL_MODEL_PATH)
        print(f"[Engine] Local ONNX Runtime loaded from: {LOCAL_MODEL_PATH}")
    except Exception as e:
        print(f"[Engine] Warning loading local ONNX: {e}")


def is_triton_available():
    try:
        res = requests.get(f"{TRITON_URL}/v2/health/ready", timeout=1.0)
        return res.status_code == 200
    except Exception:
        return False


@app.route("/")
def index():
    return render_template("index.html", triton_url=TRITON_URL, model_name=MODEL_NAME)


@app.route("/api/health")
def health():
    triton_ready = is_triton_available()
    return jsonify({
        "status": "healthy",
        "backend": "Triton Inference Server (EKS / ALB)" if triton_ready else "Local ONNX Runtime Engine",
        "triton_url": TRITON_URL,
        "triton_ready": triton_ready,
        "model_name": MODEL_NAME,
        "kserve_v2_compliant": True,
        "autoscaling_policy": "HPA (Target 70% CPU / 10 Concurrency)",
        "sla_target_ms": "< 50ms"
    })


@app.route("/api/score", methods=["POST"])
def score_transaction():
    data = request.json or {}
    amount = float(data.get("amount", 85.50))
    time_elapsed = float(data.get("time", 3600))
    v_features = data.get("features", None)

    if not v_features or len(v_features) != 28:
        # Default or semi-random PCA features
        v_features = [0.0] * 28

    full_vector = np.array([time_elapsed] + v_features + [amount], dtype=np.float32).reshape(1, 30)

    triton_ready = is_triton_available()
    t0 = time.perf_counter()

    if triton_ready:
        # KServe v2 standardized Open Inference Protocol contract
        payload = {
            "inputs": [
                {
                    "name": "transaction_features",
                    "shape": [1, 30],
                    "datatype": "FP32",
                    "data": full_vector.tolist()
                }
            ]
        }
        resp = requests.post(
            f"{TRITON_URL}/v2/models/{MODEL_NAME}/infer",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5.0
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        res_json = resp.json()
        fraud_prob = float(res_json["outputs"][0]["data"][0])
        engine = "Triton Inference Server (KServe v2)"
        contract_req = payload
        contract_resp = res_json
    else:
        # Local ONNX Engine fallback
        if LOCAL_SESSION is None:
            return jsonify({"error": "No inference engine available"}), 500
        ort_inputs = {"transaction_features": full_vector}
        ort_out = LOCAL_SESSION.run(None, ort_inputs)
        latency_ms = (time.perf_counter() - t0) * 1000
        fraud_prob = float(ort_out[0][0][0])
        engine = "Local ONNX Runtime (Triton Contract Sim)"
        contract_req = {
            "inputs": [{"name": "transaction_features", "shape": [1, 30], "datatype": "FP32", "data": full_vector.tolist()}]
        }
        contract_resp = {
            "model_name": MODEL_NAME,
            "outputs": [{"name": "fraud_probability", "shape": [1, 1], "datatype": "FP32", "data": [fraud_prob]}]
        }

    # Banking Decisioning Logic
    if fraud_prob < 0.20:
        risk_tier = "LOW_RISK"
        action = "APPROVE"
        badge_color = "#10B981" # Green
    elif fraud_prob < FRAUD_THRESHOLD:
        risk_tier = "SUSPICIOUS"
        action = "STEP_UP_2FA"
        badge_color = "#F59E0B" # Amber
    else:
        risk_tier = "HIGH_RISK_FRAUD"
        action = "DECLINE_TRANSACTION"
        badge_color = "#EF4444" # Red

    return jsonify({
        "fraud_probability": round(fraud_prob, 4),
        "risk_tier": risk_tier,
        "action": action,
        "badge_color": badge_color,
        "latency_ms": round(latency_ms, 2),
        "sla_compliant": latency_ms < 50.0,
        "inference_engine": engine,
        "kserve_request": contract_req,
        "kserve_response": contract_resp
    })


@app.route("/api/simulate-batch", methods=["POST"])
def simulate_batch():
    """Simulates dynamic batching across 20 synthetic transactions."""
    batch_size = int(request.json.get("batch_size", 20)) if request.json else 20
    batch_features = np.random.randn(batch_size, 30).astype(np.float32)

    triton_ready = is_triton_available()
    t0 = time.perf_counter()

    if triton_ready:
        payload = {
            "inputs": [
                {
                    "name": "transaction_features",
                    "shape": [batch_size, 30],
                    "datatype": "FP32",
                    "data": batch_features.tolist()
                }
            ]
        }
        resp = requests.post(f"{TRITON_URL}/v2/models/{MODEL_NAME}/infer", json=payload, timeout=5.0)
        latency_ms = (time.perf_counter() - t0) * 1000
        probs = [float(p) for p in resp.json()["outputs"][0]["data"]]
    else:
        ort_out = LOCAL_SESSION.run(None, {"transaction_features": batch_features})
        latency_ms = (time.perf_counter() - t0) * 1000
        probs = [float(p[0]) for p in ort_out[0]]

    per_item_latency = round(latency_ms / batch_size, 2)
    return jsonify({
        "batch_size": batch_size,
        "total_latency_ms": round(latency_ms, 2),
        "per_item_latency_ms": per_item_latency,
        "throughput_items_per_sec": round((batch_size / (latency_ms / 1000)), 1) if latency_ms > 0 else 0,
        "avg_fraud_prob": round(float(np.mean(probs)), 4),
        "flagged_count": sum(1 for p in probs if p >= FRAUD_THRESHOLD)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Starting Fraud Detection Portal on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
