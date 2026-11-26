"""Prediction and recommendation functions used by the course Streamlit app.

The training setup and the one-change recommendation search match the final
Colab notebook (Explainable_Method). Separate research evaluation is in Colab.
"""

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from ucimlrepo import fetch_ucirepo

SEED = 42
FEATURES = [
    "Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area",
    "Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution",
]
TARGETS = ["Heating_Load", "Cooling_Load"]
STRUCTURE = FEATURES[:5]
ADJUSTABLE = FEATURES[5:]


def load_dataset():
    dataset = fetch_ucirepo(id=242)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if X.shape != (768, 8) or y.shape != (768, 2):
        raise ValueError("Unexpected UCI dataset dimensions")
    X.columns, y.columns = FEATURES, TARGETS
    X = X.reset_index(drop=True).astype(float)
    y = y.reset_index(drop=True).astype(float)
    if X.isna().any().any() or y.isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not (X.Glazing_Area.eq(0) == X.Glazing_Area_Distribution.eq(0)).all():
        raise ValueError("Unexpected glazing encoding in source data")
    return X, y


def train_demo_model(X, y):
    # The demonstration model in the final Colab notebook is fit on all 768 rows.
    model = MultiOutputRegressor(XGBRegressor(
        n_estimators=300, max_depth=1, learning_rate=0.05,
        random_state=SEED, objective="reg:squarederror", n_jobs=1,
    ))
    model.fit(X, y)
    return model


def validate_building(building):
    b = pd.Series(building).reindex(FEATURES).astype(float)
    if b.isna().any() or not np.isfinite(b.to_numpy()).all():
        raise ValueError("Provide a valid number for each of the eight building features")
    if b["Orientation"] not in (2, 3, 4, 5):
        raise ValueError("Orientation must be 2, 3, 4, or 5")
    if b["Glazing_Area"] not in (0.0, 0.10, 0.25, 0.40):
        raise ValueError("Glazing area must be 0, 0.10, 0.25, or 0.40")
    valid_distribution = (0,) if b["Glazing_Area"] == 0 else (1, 2, 3, 4, 5)
    if b["Glazing_Area_Distribution"] not in valid_distribution:
        raise ValueError("Glazing distribution must be 0 for no glazing, otherwise 1–5")
    return b


def generate_alternatives(building, min_glazing=0.10, fixed_glazing=False):
    b = validate_building(building)
    if min_glazing not in (0.10, 0.25, 0.40):
        raise ValueError("Choose a minimum glazing of 0.10, 0.25, or 0.40")

    candidates = []
    for orientation in (2, 3, 4, 5):
        if orientation != b["Orientation"]:
            changed = b.copy()
            changed["Orientation"] = orientation
            candidates.append(changed)
    if not fixed_glazing:
        for glazing in (0.10, 0.25, 0.40):
            if glazing != b["Glazing_Area"]:
                changed = b.copy()
                changed["Glazing_Area"] = glazing
                candidates.append(changed)
    if b["Glazing_Area"] > 0:
        for distribution in (1, 2, 3, 4, 5):
            if distribution != b["Glazing_Area_Distribution"]:
                changed = b.copy()
                changed["Glazing_Area_Distribution"] = distribution
                candidates.append(changed)

    if not candidates:
        return pd.DataFrame(columns=FEATURES)
    alternatives = pd.DataFrame(candidates, columns=FEATURES)
    alternatives = alternatives[
        alternatives["Glazing_Area"].ge(min_glazing)
        & alternatives["Glazing_Area_Distribution"].isin((1, 2, 3, 4, 5))
    ]
    changed_count = alternatives[ADJUSTABLE].ne(b[ADJUSTABLE]).sum(axis=1)
    return alternatives.loc[changed_count == 1, FEATURES].drop_duplicates().reset_index(drop=True)


def recommend_design(building, model, min_glazing=0.10, fixed_glazing=False, threshold=5.0):
    """Mirror recommend_design in the final notebook."""
    b = validate_building(building)
    if b["Glazing_Area"] < min_glazing:
        return None
    alternatives = generate_alternatives(b, min_glazing, fixed_glazing)
    if alternatives.empty:
        return None
    original_prediction = model.predict(pd.DataFrame([b], columns=FEATURES))[0]
    original_total = float(original_prediction.sum())
    alternative_predictions = model.predict(alternatives[FEATURES])
    totals = alternative_predictions.sum(axis=1)
    chosen = int(np.argmin(totals))
    reduction = (original_total - float(totals[chosen])) / original_total * 100
    if reduction < threshold:
        return None
    new_design = alternatives.iloc[chosen].copy()
    return {
        "Design": new_design,
        "Changed_Feature": next(k for k in ADJUSTABLE if new_design[k] != b[k]),
        "Original_Predicted_Loads": original_prediction,
        "New_Predicted_Loads": alternative_predictions[chosen],
        "Predicted_Improvement": float(reduction),
    }


def explanation_for_change(model, original, recommended):
    """Per-recommendation SHAP differences for heating and cooling separately."""
    import shap
    pair = pd.DataFrame([validate_building(original), validate_building(recommended)],
                        index=["Original", "Recommended"])[FEATURES]
    changes = []
    for estimator in model.estimators_:
        explainer = shap.TreeExplainer(estimator)
        values = explainer(pair).values
        changes.append(values[1] - values[0])
    changes = np.asarray(changes)
    pred_before = model.predict(pair.iloc[[0]])[0]
    pred_after = model.predict(pair.iloc[[1]])[0]
    if not np.allclose(changes.sum(axis=1), pred_after - pred_before, atol=1e-3):
        raise ValueError("SHAP changes do not match the prediction difference")
    table = pd.DataFrame({
        "Feature": FEATURES,
        "Heating change": changes[0],
        "Cooling change": changes[1],
        "Combined change": changes.sum(axis=0),
    })
    return table


def match_reference(X, y, original, recommended):
    """Optional match to simulated labels; not a new accuracy assessment."""
    b1 = validate_building(original)
    b2 = validate_building(recommended)
    original_rows = X.index[X.eq(b1).all(axis=1)]
    recommended_rows = X.index[X.eq(b2).all(axis=1)]
    if len(original_rows) == 0 or len(recommended_rows) == 0:
        return None
    before = float(y.loc[original_rows[0]].sum())
    after = float(y.loc[recommended_rows[0]].sum())
    return (before - after) / before * 100
