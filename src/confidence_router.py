# confidence_router.py
# Person 3 - Confidence Routing
# Author: Anas

def route_prediction(probabilities, high_threshold=0.85, low_threshold=0.50):
    """
    Routes a prediction based on model confidence.

    Args:
        probabilities: list or tensor of [prob_normal, prob_pneumonia]
        high_threshold: above this → automated output
        low_threshold: below this → flagged for radiologist review

    Returns:
        dict with predicted class, confidence score, and routing decision
    """
    confidence = max(probabilities)
    predicted_class = probabilities.index(confidence)
    class_label = "Normal" if predicted_class == 0 else "Pneumonia"

    if confidence >= high_threshold:
        decision = "Automated"
    elif confidence < low_threshold:
        decision = "Review"
    else:
        decision = "Review"  # middle ground also goes to review

    return {
        "predicted_class": class_label,
        "confidence": round(confidence, 4),
        "decision": decision
    }