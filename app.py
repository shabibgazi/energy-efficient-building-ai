"""Interactive building energy-load prediction and design recommendations."""

import numpy as np
import pandas as pd
import streamlit as st
from ucimlrepo import fetch_ucirepo
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

st.set_page_config(page_title="Building Energy Design", page_icon="🏠", layout="wide")

FEATURES = [
    "Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area",
    "Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution"
]
STRUCTURE = FEATURES[:5]
ADJUSTABLE = FEATURES[5:]
FRIENDLY = {
    "Relative_Compactness": "Relative compactness", "Surface_Area": "Surface area",
    "Wall_Area": "Wall area", "Roof_Area": "Roof area", "Overall_Height": "Overall height",
    "Orientation": "Orientation", "Glazing_Area": "Glazing area",
    "Glazing_Area_Distribution": "Glazing distribution"
}

@st.cache_data(show_spinner="Loading the UCI Energy Efficiency dataset…")
def load_data():
    energy = fetch_ucirepo(id=242)
    X = energy.data.features.copy()
    y = energy.data.targets.copy()
    if X.shape != (768, 8) or y.shape != (768, 2):
        raise ValueError("The source data has changed shape; expected 768 rows, 8 inputs, and 2 targets.")
    X.columns = FEATURES
    y.columns = ["Heating_Load", "Cooling_Load"]
    X = X.reset_index(drop=True).astype(float)
    y = y.reset_index(drop=True).astype(float)
    return X, y

@st.cache_resource(show_spinner="Training the recommendation model…")
def train_model(X, y):
    model = MultiOutputRegressor(XGBRegressor(
        n_estimators=300, max_depth=1, learning_rate=0.05,
        random_state=42, objective="reg:squarederror", n_jobs=1
    ))
    model.fit(X, y)
    return model


def generate_alternatives(building, minimum_glazing=0.10, fixed_glazing=False):
    """Change exactly one editable feature per candidate, retaining structural inputs."""
    b = building.copy()
    candidates = []
    for orientation in (2, 3, 4, 5):
        if orientation != b["Orientation"]:
            new = b.copy()
            new["Orientation"] = orientation
            candidates.append(new)
    if not fixed_glazing:
        for glazing in (0.10, 0.25, 0.40):
            if glazing != b["Glazing_Area"]:
                new = b.copy()
                new["Glazing_Area"] = glazing
                candidates.append(new)
    if b["Glazing_Area"] > 0:
        for distribution in (1, 2, 3, 4, 5):
            if distribution != b["Glazing_Area_Distribution"]:
                new = b.copy()
                new["Glazing_Area_Distribution"] = distribution
                candidates.append(new)
    if not candidates:
        return pd.DataFrame(columns=FEATURES)
    alternatives = pd.DataFrame(candidates, columns=FEATURES)
    alternatives = alternatives[
        (alternatives["Glazing_Area"] >= minimum_glazing)
        & alternatives["Glazing_Area_Distribution"].isin([1, 2, 3, 4, 5])
    ]
    counts = alternatives[ADJUSTABLE].ne(b[ADJUSTABLE]).sum(axis=1)
    return alternatives.loc[counts == 1, FEATURES].drop_duplicates().reset_index(drop=True)


def recommend(building, model, minimum_glazing=0.10, fixed_glazing=False, threshold=5.0):
    original_prediction = model.predict(pd.DataFrame([building], columns=FEATURES))[0]
    if building["Glazing_Area"] < minimum_glazing:
        return original_prediction, None
    alternatives = generate_alternatives(building, minimum_glazing, fixed_glazing)
    if alternatives.empty:
        return original_prediction, None
    alternatives_predicted = model.predict(alternatives)
    index = int(np.argmin(alternatives_predicted.sum(axis=1)))
    proposed = alternatives.iloc[index].copy()
    updated_prediction = alternatives_predicted[index]
    reduction = (original_prediction.sum() - updated_prediction.sum()) / original_prediction.sum() * 100
    if reduction < threshold:
        return original_prediction, None
    changed_feature = next(key for key in ADJUSTABLE if proposed[key] != building[key])
    return original_prediction, (proposed, updated_prediction, reduction, changed_feature)


def explain_one(model, building, target_index):
    """Return per-feature SHAP values for one model prediction."""
    import shap
    explainer = shap.TreeExplainer(model.estimators_[target_index])
    values = explainer(pd.DataFrame([building], columns=FEATURES)).values[0]
    return pd.DataFrame({
        "Feature": [FRIENDLY[f] for f in FEATURES],
        "SHAP contribution": values
    }).sort_values("SHAP contribution")


st.title("🏠 Energy-Efficient Building Design")
st.write("Estimate a residential building's heating and cooling loads, compare possible design changes, and inspect the model's predictions.")
st.caption("Course research prototype · Shabib Hyder Gazi & Satnam Singh Khokhar · Georgia State University")

try:
    X, y = load_data()
    model = train_model(X, y)
except Exception as exc:
    st.error("The dataset or model could not be loaded. Check the internet connection and try refreshing the app.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

structures = X[STRUCTURE].drop_duplicates().reset_index(drop=True)
with st.sidebar:
    st.header("Building inputs")
    st.write("Choose a structure from the research dataset, then adjust its orientation and windows.")
    structure_id = st.selectbox(
        "Building geometry",
        list(range(len(structures))),
        format_func=lambda i: f"Structure {i + 1} · compactness {structures.loc[i, 'Relative_Compactness']:.2f}"
    )
    structure = structures.loc[structure_id]
    st.caption("Fixed structural characteristics")
    for feature in STRUCTURE:
        st.text(f"{FRIENDLY[feature]}: {structure[feature]:g}")
    st.divider()
    orientation = st.selectbox("Orientation (dataset code)", [2, 3, 4, 5], index=0)
    glazing = st.select_slider("Glazing area fraction", options=[0.0, 0.10, 0.25, 0.40], value=0.40)
    distribution_choices = [0] if glazing == 0 else [1, 2, 3, 4, 5]
    distribution = st.selectbox("Glazing distribution (dataset code)", distribution_choices)
    st.divider()
    st.subheader("Recommendation settings")
    fixed_glazing = st.checkbox("Keep glazing area unchanged", value=False)
    min_glazing = st.selectbox("Minimum glazing in recommended design", [0.10, 0.25, 0.40], index=0)
    threshold = st.slider("Minimum predicted improvement (%)", 0.0, 15.0, 5.0, 0.5)

building = structure.copy()
building["Orientation"] = float(orientation)
building["Glazing_Area"] = float(glazing)
building["Glazing_Area_Distribution"] = float(distribution)
building = building.reindex(FEATURES)

original, recommendation = recommend(building, model, min_glazing, fixed_glazing, threshold)

st.subheader("Predicted energy loads")
col1, col2, col3 = st.columns(3)
col1.metric("Heating load", f"{original[0]:.2f} kWh/m²")
col2.metric("Cooling load", f"{original[1]:.2f} kWh/m²")
col3.metric("Combined load", f"{original.sum():.2f} kWh/m²")

st.subheader("Recommended design")
if recommendation is None:
    if glazing < min_glazing:
        st.info("The input design is below the chosen minimum glazing level. Choose a compatible setting to evaluate recommendations.")
    else:
        st.info(f"No permitted single-feature change meets the {threshold:g}% predicted-improvement threshold.")
else:
    suggested, updated, improvement, changed = recommendation
    st.success(f"Suggested change: {FRIENDLY[changed]} · {building[changed]:g} → {suggested[changed]:g}")
    st.metric("Estimated reduction in combined load", f"{improvement:.2f}%")
    comparison = pd.DataFrame({
        "Current design": [building[f] for f in ADJUSTABLE],
        "Suggested design": [suggested[f] for f in ADJUSTABLE]
    }, index=[FRIENDLY[f] for f in ADJUSTABLE])
    st.dataframe(comparison, use_container_width=True)
    chart = pd.DataFrame({
        "Original": [original[0], original[1], original.sum()],
        "Suggested": [updated[0], updated[1], updated.sum()]
    }, index=["Heating", "Cooling", "Combined"])
    st.bar_chart(chart, y_label="Predicted load (kWh/m²)")

st.subheader("Why did the model make this prediction?")
st.write("SHAP scores show how each input affected the model's output. They describe the fitted model, not a physical cause-and-effect relationship.")
target = st.radio("Explain", ["Heating", "Cooling"], horizontal=True)
try:
    values = explain_one(model, building, 0 if target == "Heating" else 1)
    st.bar_chart(values.set_index("Feature"), horizontal=True, x_label="SHAP contribution (kWh/m²)")
    with st.expander("View SHAP values"):
        st.dataframe(values, use_container_width=True, hide_index=True)
except Exception as exc:
    st.warning("The SHAP explanation is unavailable in this environment.")
    with st.expander("Technical details"):
        st.code(str(exc))

st.divider()
st.caption("The model is trained on 768 simulated UCI Energy Efficiency designs. Results are research estimates, not measured energy savings or construction recommendations. Predictions for unfamiliar building geometries can be less reliable.")
