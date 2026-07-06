import pandas as pd

EXPLANATION_FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]


def explain_recommendation(
    raw_profile,
    crop,
    baseline_stats,
):
    explanation = []

    for feature in EXPLANATION_FEATURES:

        value = raw_profile[feature]

        mean = baseline_stats.loc[crop, (feature, "mean")]
        std = baseline_stats.loc[crop, (feature, "std")]

        lower = mean - std
        upper = mean + std

        if value < lower:
            status = "Below Typical Range"

        elif value > upper:
            status = "Above Typical Range"

        else:
            status = "Within Typical Range"

        explanation.append(
            {
                "Feature": feature.capitalize(),
                "Observed": round(value, 2),
                "Typical Lower": round(lower, 2),
                "Typical Upper": round(upper, 2),
                "Status": status,
            }
        )

    return pd.DataFrame(explanation)


def generate_recommendations(
    raw_profile,
    pipeline,
    baseline_stats,
    top_n=3,
):
    input_df = pd.DataFrame([raw_profile])

    probabilities = pipeline.predict_proba(input_df)[0]

    recommendations = (
        pd.DataFrame(
            {
                "Crop": pipeline.classes_,
                "Probability": probabilities,
            }
        )
        .sort_values(
            "Probability",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    explanations = {}

    for crop in recommendations["Crop"]:
        explanations[crop] = explain_recommendation(
            raw_profile,
            crop,
            baseline_stats,
        )

    return recommendations, explanations