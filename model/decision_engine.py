import pandas as pd

from model.predict import predict_scheduler
from drift.workload_detector import detect_drift


def make_decision(df):
    """
    Combine AI scheduler prediction and workload drift detection.
    """

    # AI scheduler prediction
    prediction, confidence, features = predict_scheduler(df)

    # Drift detection
    drift_result = detect_drift(df)

    drift_status = drift_result["status"]
    drift_score = drift_result["drift_score"]
    trust_ai = drift_result["trust_ai"]

    # Highest prediction confidence
    ai_confidence = max(confidence.values())

    # Decision logic
    if not trust_ai:
        decision = "FALLBACK"
        selected_scheduler = None
        reason = "Workload drift detected. AI recommendation is not trusted."

    elif ai_confidence < 0.50:
        decision = "RE-EVALUATE"
        selected_scheduler = None
        reason = "Workload is stable, but AI confidence is low."

    else:
        decision = "TRUST AI"
        selected_scheduler = prediction
        reason = "Workload is stable and AI confidence is sufficient."

    return {
        "ai_prediction": prediction,
        "confidence": confidence,
        "ai_confidence": ai_confidence,
        "drift_status": drift_status,
        "drift_score": drift_score,
        "trust_ai": trust_ai,
        "decision": decision,
        "selected_scheduler": selected_scheduler,
        "reason": reason,
        "features": features
    }


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("ADAPTIVE AI SCHEDULER - DECISION ENGINE")
    print("=" * 60)

    df = pd.read_csv("process_data.csv")

    # Test with first 10 processes
    workload = df.head(10)

    result = make_decision(workload)

    print(f"\nAI Recommendation : {result['ai_prediction']}")
    print(f"AI Confidence     : {result['ai_confidence'] * 100:.2f}%")
    print(f"Drift Status      : {result['drift_status']}")
    print(f"Drift Score       : {result['drift_score']:.4f}")

    print("\n" + "-" * 60)

    print(f"Decision           : {result['decision']}")

    if result["selected_scheduler"]:
        print(f"Selected Scheduler : {result['selected_scheduler']}")

    print(f"Reason             : {result['reason']}")

    print("\nScheduler Confidence:")

    for scheduler, probability in result["confidence"].items():
        print(f"{scheduler}: {probability * 100:.2f}%")