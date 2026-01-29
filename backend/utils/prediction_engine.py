import random

# When you add your model, load it here:
MODEL = None     # placeholder
MODEL_LOADED = False


def predict_risk(features):
    """
    Returns mock prediction for now:
    - low
    - moderate
    - high

    Later:
      - plug your TensorFlow model here
    """

    # MOCK PREDICTION (random)
    risk_levels = ["low", "moderate", "high"]
    risk = random.choice(risk_levels)

    # mock probability
    score = round(random.uniform(0.1, 0.9), 3)

    return risk, score
