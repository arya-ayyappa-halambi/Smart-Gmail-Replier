"""
evaluate.py - AIML Skill Lab Evaluation Benchmark for Smart Gmail Replier.

Runs benchmark evaluation on a held-out test set (data/test_emails.json):
  1. Computes Intent Classification Accuracy, Precision, Recall, and F1-Scores.
  2. Computes Urgency Classification Accuracy and Distribution.
  3. Measures per-sample and average inference latency (in milliseconds).
  4. Evaluates Style & Context RAG retrieval relevance (top-1 and top-2 similarity scores).
  5. Displays a formatted terminal report with ASCII/Unicode tables.
  6. Exports full benchmark results to evaluation_report.json.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from tabulate import tabulate

# Ensure parent directory is in python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import ml_pipeline

TEST_DATA_PATH = BASE_DIR / "data" / "test_emails.json"
REPORT_OUTPUT_PATH = BASE_DIR / "evaluation_report.json"


def run_evaluation() -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("      🚀 AIML SKILL LAB: SMART GMAIL REPLIER BENCHMARK EVALUATION")
    print("=" * 75)
    print(f"Loading held-out test dataset from: {TEST_DATA_PATH.resolve()}")

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}")

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    total_samples = len(test_samples)
    print(f"Total test instances to evaluate: {total_samples}")
    print("-" * 75)

    # Warm up pipeline and models
    print("[1/4] Initializing ML classifier and warming up RAG index...")
    _ = ml_pipeline.analyze_email("Warmup", "test@bmsce.ac.in", "Warmup content")

    y_true_intent: List[str] = []
    y_pred_intent: List[str] = []
    intent_confidences: List[float] = []

    y_true_urgency: List[str] = []
    y_pred_urgency: List[str] = []
    urgency_confidences: List[float] = []

    latencies_ms: List[float] = []
    rag_top1_sims: List[float] = []
    rag_top2_sims: List[float] = []

    detailed_records: List[Dict[str, Any]] = []

    print("[2/4] Executing inference across all test samples...")
    for idx, item in enumerate(test_samples):
        sub = item.get("subject", "")
        sender = item.get("sender", "")
        body = item.get("body", "")
        exp_intent = item.get("expected_intent") or item.get("intent")
        exp_urgency = item.get("expected_urgency") or item.get("urgency")

        # Measure precise inference latency
        t_start = time.perf_counter()
        result = ml_pipeline.analyze_email(sub, sender, body)
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000

        pred_intent = result["intent"]
        intent_conf = result["intent_confidence"]
        pred_urgency = result["urgency"]
        urgency_conf = result["urgency_confidence"]
        rag_context = result.get("rag_context", [])

        # Record metrics
        y_true_intent.append(exp_intent)
        y_pred_intent.append(pred_intent)
        intent_confidences.append(intent_conf)

        if exp_urgency:
            y_true_urgency.append(exp_urgency)
            y_pred_urgency.append(pred_urgency)
            urgency_confidences.append(urgency_conf)

        latencies_ms.append(t_elapsed_ms)

        top1_sim = rag_context[0]["similarity"] if len(rag_context) > 0 else 0.0
        top2_sim = rag_context[1]["similarity"] if len(rag_context) > 1 else 0.0
        rag_top1_sims.append(top1_sim)
        rag_top2_sims.append(top2_sim)

        detailed_records.append({
            "id": item.get("id", f"sample_{idx+1}"),
            "subject": sub[:40] + "..." if len(sub) > 40 else sub,
            "expected_intent": exp_intent,
            "pred_intent": pred_intent,
            "intent_match": exp_intent == pred_intent,
            "intent_confidence": intent_conf,
            "expected_urgency": exp_urgency,
            "pred_urgency": pred_urgency,
            "urgency_match": exp_urgency == pred_urgency if exp_urgency else None,
            "urgency_confidence": urgency_conf,
            "latency_ms": round(t_elapsed_ms, 2),
            "rag_top1_score": top1_sim,
        })

    # Compute aggregate metrics
    intent_accuracy = float(accuracy_score(y_true_intent, y_pred_intent))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, average="weighted", zero_division=0
    )

    clf_report_dict = classification_report(
        y_true_intent, y_pred_intent, output_dict=True, zero_division=0
    )

    urgency_accuracy = (
        float(accuracy_score(y_true_urgency, y_pred_urgency))
        if y_true_urgency else 0.0
    )

    avg_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    avg_top1_sim = float(np.mean(rag_top1_sims))
    avg_top2_sim = float(np.mean(rag_top2_sims))
    avg_confidence = float(np.mean(intent_confidences))

    print("[3/4] Compiling Benchmark Evaluation Summary Tables...\n")

    # Table 1: Per-Sample Results Table
    sample_rows = []
    for r in detailed_records:
        intent_status = "✅" if r["intent_match"] else "❌"
        urgency_status = "✅" if r["urgency_match"] else "❌"
        sample_rows.append([
            r["id"],
            r["subject"],
            f"{r['pred_intent']} {intent_status}",
            f"{r['intent_confidence']:.2f}",
            f"{r['pred_urgency']} {urgency_status}",
            f"{r['rag_top1_score']:.3f}",
            f"{r['latency_ms']:.1f} ms",
        ])

    print(tabulate(
        sample_rows,
        headers=["ID", "Subject", "Pred Intent", "Conf", "Pred Urgency", "RAG Sim", "Latency"],
        tablefmt="fancy_grid",
    ))
    print()

    # Table 2: Per-Class Intent Metrics
    per_class_rows = []
    for cls_name in ml_pipeline.INTENT_CLASSES:
        if cls_name in clf_report_dict:
            stats = clf_report_dict[cls_name]
            per_class_rows.append([
                cls_name,
                f"{stats['precision']:.3f}",
                f"{stats['recall']:.3f}",
                f"{stats['f1-score']:.3f}",
                int(stats["support"]),
            ])

    print("📊 PER-CLASS INTENT PERFORMANCE")
    print(tabulate(
        per_class_rows,
        headers=["Intent Category", "Precision", "Recall", "F1-Score", "Support"],
        tablefmt="rounded_grid",
    ))
    print()

    # Table 3: Executive Summary
    summary_metrics = [
        ["Intent Classification Accuracy", f"{intent_accuracy * 100:.2f}%"],
        ["Intent Macro F1-Score", f"{f1_macro:.4f}"],
        ["Intent Weighted F1-Score", f"{f1_weighted:.4f}"],
        ["Urgency Classification Accuracy", f"{urgency_accuracy * 100:.2f}%"],
        ["Mean Model Confidence", f"{avg_confidence:.4f}"],
        ["Average Inference Latency", f"{avg_latency:.2f} ms"],
        ["95th Percentile Latency (p95)", f"{p95_latency:.2f} ms"],
        ["RAG Mean Top-1 Cosine Similarity", f"{avg_top1_sim:.4f}"],
        ["RAG Mean Top-2 Cosine Similarity", f"{avg_top2_sim:.4f}"],
    ]

    print("🏆 SYSTEM BENCHMARK SUMMARY")
    print(tabulate(summary_metrics, headers=["Metric", "Result"], tablefmt="heavy_grid"))
    print()

    # Export report to JSON
    report_data = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_evaluated": str(TEST_DATA_PATH.resolve()),
        "total_test_samples": total_samples,
        "metrics": {
            "intent_accuracy": round(intent_accuracy, 4),
            "intent_macro_f1": round(f1_macro, 4),
            "intent_weighted_f1": round(f1_weighted, 4),
            "urgency_accuracy": round(urgency_accuracy, 4),
            "mean_confidence": round(avg_confidence, 4),
            "average_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "rag_mean_top1_similarity": round(avg_top1_sim, 4),
            "rag_mean_top2_similarity": round(avg_top2_sim, 4),
        },
        "per_class_intent_metrics": clf_report_dict,
        "per_sample_details": detailed_records,
    }

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as rf:
        json.dump(report_data, rf, indent=2)

    print(f"[4/4] ✅ Benchmark report successfully saved to: {REPORT_OUTPUT_PATH.resolve()}\n")
    return report_data


if __name__ == "__main__":
    run_evaluation()
