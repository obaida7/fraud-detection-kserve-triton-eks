#!/usr/bin/env python3
"""
triton_client.py
Production-grade KServe v2 / Triton Open Inference Protocol Client.
Supports:
1. HTTP (/v2/models/{model_name}/infer) using standard REST and JSON contracts.
2. gRPC for ultra-low sub-millisecond inference latency.
3. Health check probes (/v2/health/ready and /v2/health/live).
4. Dynamic banking risk-tiering and threshold enforcement.
"""

import sys
import json
import time
import argparse
import numpy as np
import requests

# Try importing tritonclient if available
try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    TRITON_CLIENT_LIB_AVAILABLE = True
except ImportError:
    TRITON_CLIENT_LIB_AVAILABLE = False


class TritonInferenceClient:
    def __init__(self, host: str = "localhost", http_port: int = 8000, grpc_port: int = 8001, model_name: str = "fraud_detection"):
        self.host = host.replace("http://", "").replace("https://", "").rstrip("/")
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.model_name = model_name
        self.base_url = f"http://{self.host}:{self.http_port}"
        self.fraud_threshold = 0.35

    def check_health(self) -> dict:
        """Checks Triton server readiness and liveness endpoints."""
        live_url = f"{self.base_url}/v2/health/live"
        ready_url = f"{self.base_url}/v2/health/ready"
        model_ready_url = f"{self.base_url}/v2/models/{self.model_name}/ready"

        status = {}
        try:
            r_live = requests.get(live_url, timeout=3)
            status["server_live"] = r_live.status_code == 200
        except Exception as e:
            status["server_live"] = False
            status["live_error"] = str(e)

        try:
            r_ready = requests.get(ready_url, timeout=3)
            status["server_ready"] = r_ready.status_code == 200
        except Exception as e:
            status["server_ready"] = False
            status["ready_error"] = str(e)

        try:
            r_model = requests.get(model_ready_url, timeout=3)
            status["model_ready"] = r_model.status_code == 200
        except Exception as e:
            status["model_ready"] = False
            status["model_error"] = str(e)

        return status

    def get_model_metadata(self) -> dict:
        """Retrieves model metadata and standardized I/O tensor contract."""
        url = f"{self.base_url}/v2/models/{self.model_name}"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        return res.json()

    def predict_kserve_v2_raw(self, features: np.ndarray) -> dict:
        """
        Sends prediction using raw KServe v2 / Triton Open Inference Protocol JSON format.
        Tensor shape: [batch_size, num_features]
        """
        if features.ndim == 1:
            features = np.expand_dims(features, axis=0)

        batch_size, num_features = features.shape
        payload = {
            "inputs": [
                {
                    "name": "transaction_features",
                    "shape": [batch_size, num_features],
                    "datatype": "FP32",
                    "data": features.astype(np.float32).tolist()
                }
            ]
        }

        url = f"{self.base_url}/v2/models/{self.model_name}/infer"
        t0 = time.perf_counter()
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        latency_ms = (time.perf_counter() - t0) * 1000

        if response.status_code != 200:
            raise RuntimeError(f"Triton returned HTTP {response.status_code}: {response.text}")

        res_json = response.json()
        raw_probs = res_json["outputs"][0]["data"]

        results = []
        for prob in raw_probs:
            if prob < 0.20:
                tier = "LOW_RISK"
                action = "APPROVE"
            elif prob < self.fraud_threshold:
                tier = "SUSPICIOUS"
                action = "CHALLENGE_2FA"
            else:
                tier = "HIGH_RISK_FRAUD"
                action = "DECLINE"

            results.append({
                "fraud_probability": round(float(prob), 4),
                "risk_tier": tier,
                "action": action
            })

        return {
            "model_name": self.model_name,
            "latency_ms": round(latency_ms, 2),
            "batch_size": batch_size,
            "predictions": results
        }

    def predict_grpc(self, features: np.ndarray) -> dict:
        """Sends prediction via Triton gRPC client for minimal wire latency."""
        if not TRITON_CLIENT_LIB_AVAILABLE:
            raise ImportError("tritonclient library is not installed for gRPC mode.")

        if features.ndim == 1:
            features = np.expand_dims(features, axis=0)

        t0 = time.perf_counter()
        with grpcclient.InferenceServerClient(url=f"{self.host}:{self.grpc_port}") as client:
            inputs = [
                grpcclient.InferInput("transaction_features", features.shape, "FP32")
            ]
            inputs[0].set_data_from_numpy(features.astype(np.float32))

            outputs = [
                grpcclient.InferRequestedOutput("fraud_probability")
            ]

            response = client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
            latency_ms = (time.perf_counter() - t0) * 1000

            raw_probs = response.as_numpy("fraud_probability").flatten().tolist()

            results = []
            for prob in raw_probs:
                results.append({
                    "fraud_probability": round(float(prob), 4),
                    "risk_tier": "HIGH_RISK_FRAUD" if prob >= self.fraud_threshold else "LOW_RISK",
                    "action": "DECLINE" if prob >= self.fraud_threshold else "APPROVE"
                })

            return {
                "protocol": "gRPC",
                "latency_ms": round(latency_ms, 2),
                "predictions": results
            }


def main():
    parser = argparse.ArgumentParser(description="Triton / KServe v2 Inference Client")
    parser.add_argument("--host", type=str, default="localhost", help="Triton host / ALB DNS")
    parser.add_argument("--http_port", type=int, default=8000, help="HTTP port (default 8000)")
    parser.add_argument("--grpc_port", type=int, default=8001, help="gRPC port (default 8001)")
    parser.add_argument("--model", type=str, default="fraud_detection", help="Model name")
    parser.add_argument("--mode", type=str, choices=["health", "metadata", "predict", "grpc"], default="predict")
    parser.add_argument("--samples", type=int, default=1, help="Number of simulated transactions to score")
    args = parser.parse_args()

    client = TritonInferenceClient(host=args.host, http_port=args.http_port, grpc_port=args.grpc_port, model_name=args.model)

    if args.mode == "health":
        print(f"Checking health at {client.base_url}...")
        status = client.check_health()
        print(json.dumps(status, indent=2))
        return

    if args.mode == "metadata":
        print(f"Fetching metadata for '{args.model}'...")
        metadata = client.get_model_metadata()
        print(json.dumps(metadata, indent=2))
        return

    # Generate synthetic transaction features for testing
    print(f"\nScoring {args.samples} transaction(s) using KServe v2 standardized contract...")
    dummy_features = np.random.randn(args.samples, 30).astype(np.float32)

    if args.mode == "grpc":
        res = client.predict_grpc(dummy_features)
    else:
        res = client.predict_kserve_v2_raw(dummy_features)

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
