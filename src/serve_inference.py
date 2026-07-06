from pathlib import Path
import joblib
import pandas as pd
import pickle
import sys

# ==========================================================
# Setup
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rhea_features import AgronomicFeatureEngineer
from src.inference import generate_recommendations

# ==========================================================
# Farmer Input
# ==========================================================

def get_farmer_input():
    """
    Collect measurements from the user.
    Press Enter to use the default example value.
    """

    print("\nEnter Farmer Measurements")
    print("(Press Enter to use the default value)\n")

    defaults = {
        "N": 85,
        "P": 40,
        "K": 50,
        "temperature": 25,
        "humidity": 80,
        "ph": 6.5,
        "rainfall": 120,
    }

    farmer_input = {}

    for feature, default in defaults.items():

        value = input(f"{feature:<15} [{default}]: ")

        if value.strip() == "":
            farmer_input[feature] = default
        else:
            farmer_input[feature] = float(value)

    return farmer_input
# ==========================================================
# Compatibility Loader
# ==========================================================

def read_pickle_compat(path):
    try:
        return pd.read_pickle(path)

    except Exception:

        aliases = {
            "numpy._core.numeric": "numpy.core.numeric",
            "numpy._core.multiarray": "numpy.core.multiarray",
            "numpy._core.umath": "numpy.core.umath",
        }

        for old, new in aliases.items():
            if old not in sys.modules:
                sys.modules[old] = __import__(new, fromlist=[None])

        class CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):

                if module.startswith("numpy._core"):
                    module = module.replace(
                        "numpy._core",
                        "numpy.core"
                    )

                return super().find_class(module, name)

        with open(path, "rb") as f:
            return CompatUnpickler(f).load()


# ==========================================================
# Load Production Artifacts
# ==========================================================

MODEL_PATH = ROOT / "models" / "rhea_knn_recommendation_pipeline.joblib"
BASELINE_PATH = ROOT / "models" / "crop_baselines.pkl"

pipeline = joblib.load(MODEL_PATH)
crop_baselines = read_pickle_compat(BASELINE_PATH)


# ==========================================================
# Farmer Input
# ==========================================================

farmer_input = get_farmer_input()


# ==========================================================
# Run Inference
# ==========================================================

recommendations, explanations = generate_recommendations(
    raw_profile=farmer_input,
    pipeline=pipeline,
    baseline_stats=crop_baselines,
    top_n=3
)


# ==========================================================
# Pretty Report
# ==========================================================

print("\n")
print("=" * 80)
print("               RHEA AGRICULTURAL RECOMMENDATION ENGINE")
print("=" * 80)

print("\nDEPLOYMENT STATUS")
print("-" * 80)
print("Model                : K-Nearest Neighbors")
print("Pipeline             : Loaded Successfully")
print("Inference            : Completed")
print("Recommendation Count : 3")


print("\nFARMER INPUT")
print("-" * 80)

for feature, value in farmer_input.items():
    print(f"{feature:<15} : {value}")


print("\nTOP RECOMMENDATIONS")
print("-" * 80)

for rank, row in recommendations.iterrows():

    confidence = row["Probability"] * 100

    print(
        f"{rank+1}. "
        f"{row['Crop'].capitalize():15}"
        f"{confidence:6.2f}%"
    )


top_crop = recommendations.iloc[0]["Crop"]

print("\n")
print("=" * 80)
print(f"WHY '{top_crop.upper()}'?")
print("=" * 80)

print(
    explanations[top_crop].to_string(index=False)
)


print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
Recommended Crop : {top_crop.capitalize()}

Confidence       : {recommendations.iloc[0]['Probability']*100:.2f}%

The recommendation was produced by the complete production
pipeline, including:

✓ AgronomicFeatureEngineer
✓ MinMaxScaler
✓ KNN Classifier

Every transformation applied during training was reused
automatically during inference.

This demonstrates that the serialized deployment artifact
can receive raw farmer measurements and generate
production-ready crop recommendations without any manual
feature engineering.
""")