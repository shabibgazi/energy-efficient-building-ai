"""Streamlit front end for the residential building design research prototype."""

import numpy as np
import pandas as pd
import streamlit as st

from model import (
    FEATURES, STRUCTURE, ADJUSTABLE, TARGETS, load_dataset, train_demo_model,
    validate_building, recommend_design, explanation_for_change, match_reference,
)

st.set_page_config(page_title="Building Energy Design | AI Research", page_icon="🏠", layout="wide")

st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1170px;}
h1, h2, h3 {letter-spacing: -0.025em;}
[data-testid="stMetric"] {padding: 0.8rem 1rem; border-radius: 0.65rem; border: 1px solid #d7dfe4;}
</style>""", unsafe_allow_html=True)

NAMES = {
    "Relative_Compactness": "Relative compactness",
    "Surface_Area": "Surface area (m²)",
    "Wall_Area": "Wall area (m²)",
    "Roof_Area": "Roof area (m²)",
    "Overall_Height": "Overall height (m)",
    "Orientation": "Orientation (code)",
    "Glazing_Area": "Glazing area (fraction)",
    "Glazing_Area_Distribution": "Glazing distribution (code)",
}

@st.cache_data(show_spinner="Downloading the research dataset…")
def get_data():
    return load_dataset()

@st.cache_resource(show_spinner="Training the prediction model…")
def get_model():
    # Use the same training setup as the notebook's Section 10 demonstration.
    X, y = get_data()
    return train_demo_model(X, y)

st.title("Energy-Efficient Building Design Recommender")
st.write("Explore how changes to a residential building design affect predicted heating and cooling requirements.")
st.caption("AI course research project · Shabib Hyder Gazi & Satnam Singh Khokhar · Georgia State University")

try:
    X, y = get_data()
    model = get_model()
except Exception as exc:
    st.error("The application could not load its dataset or model. Please refresh the page and try again.")
    with st.expander("Error details"):
        st.code(str(exc))
    st.stop()

structures = X[STRUCTURE].drop_duplicates().reset_index(drop=True)

with st.sidebar:
    st.header("Building settings")
    st.caption("Start from a simulated building structure, or enter your own structural values.")
    source = st.radio("Structure input", ["Choose a dataset structure", "Enter structural values"], index=0)
    if source == "Choose a dataset structure":
        sid = st.selectbox("Structure", range(len(structures)),
                           format_func=lambda i: f"Structure {i + 1} · compactness {structures.iloc[i]['Relative_Compactness']:.2f}")
        structure = structures.iloc[sid].copy()
        with st.expander("View structural characteristics", expanded=False):
            for f in STRUCTURE:
                st.write(f"{NAMES[f]}: {structure[f]:g}")
    else:
        st.caption("Values outside the training dataset may produce less reliable predictions.")
        defaults = structures.iloc[0]
        vals = {}
        for f in STRUCTURE:
            vals[f] = st.number_input(NAMES[f], value=float(defaults[f]), format="%.3f",
                                       min_value=0.001, step=0.1)
        structure = pd.Series(vals)

    st.divider()
    st.subheader("Editable features")
    orientation = st.selectbox("Orientation", (2, 3, 4, 5), index=0,
                                help="Dataset categories rather than geographic compass degrees")
    glazing = st.select_slider("Glazing area", options=(0.0, 0.10, 0.25, 0.40), value=0.40)
    dist_options = (0,) if glazing == 0 else (1, 2, 3, 4, 5)
    distribution = st.selectbox("Glazing distribution", dist_options,
                                 help="Code 0 applies to no glazing; 1–5 are dataset distribution categories")

    st.divider()
    st.subheader("Recommendation constraints")
    fixed_glazing = st.checkbox("Keep glazing area unchanged", value=False)
    min_glazing = st.selectbox("Minimum permitted glazing", (0.10, 0.25, 0.40))
    threshold = st.slider("Minimum predicted improvement (%)", min_value=0.0,
                          max_value=20.0, value=5.0, step=0.5)

building = structure.copy()
building["Orientation"] = float(orientation)
building["Glazing_Area"] = float(glazing)
building["Glazing_Area_Distribution"] = float(distribution)

try:
    building = validate_building(building)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

current = model.predict(pd.DataFrame([building], columns=FEATURES))[0]

st.subheader("1. Predicted energy loads")
a, b, c = st.columns(3)
a.metric("Heating load", f"{current[0]:.2f} kWh/m²")
b.metric("Cooling load", f"{current[1]:.2f} kWh/m²")
c.metric("Combined load", f"{current.sum():.2f} kWh/m²")

with st.expander("View all eight input values"):
    input_table = pd.DataFrame({"Feature": [NAMES[f] for f in FEATURES],
                                "Value": [building[f] for f in FEATURES]})
    st.dataframe(input_table, hide_index=True, use_container_width=True)

st.subheader("2. Recommended design")
recommendation = recommend_design(building, model, min_glazing=min_glazing,
                                  fixed_glazing=fixed_glazing, threshold=threshold)

if recommendation is None:
    if glazing < min_glazing:
        st.info("The input building does not satisfy the minimum glazing constraint. Choose compatible values to continue.")
    else:
        st.info(f"No permitted one-feature change meets the {threshold:g}% predicted-improvement threshold for this building.")
else:
    chosen = recommendation["Design"]
    updated = recommendation["New_Predicted_Loads"]
    change_feature = recommendation["Changed_Feature"]
    improvement = recommendation["Predicted_Improvement"]

    st.success(f"Recommended change: {NAMES[change_feature]} · {building[change_feature]:g} → {chosen[change_feature]:g}")
    st.metric("Predicted combined-load reduction", f"{improvement:.2f}%")

    differences = pd.DataFrame({
        "Design characteristic": [NAMES[f] for f in ADJUSTABLE],
        "Original": [building[f] for f in ADJUSTABLE],
        "Recommended": [chosen[f] for f in ADJUSTABLE],
    })
    st.dataframe(differences, hide_index=True, use_container_width=True)

    comparison = pd.DataFrame({"Original": [current[0], current[1]],
                               "Recommended": [updated[0], updated[1]]},
                              index=["Heating", "Cooling"])
    st.bar_chart(comparison, y_label="Predicted load (kWh/m²)", horizontal=False)
    with st.expander("View exact predicted loads"):
        expanded = comparison.copy()
        expanded.loc["Combined"] = [current.sum(), updated.sum()]
        expanded["Reduction"] = expanded["Original"] - expanded["Recommended"]
        st.dataframe(expanded.round(3), use_container_width=True)

    st.subheader("3. Why does the model recommend this change?")
    st.write("SHAP shows how feature contributions differ between the original and recommended predictions.")
    try:
        shap_table = explanation_for_change(model, building, chosen)
        relevant = shap_table.loc[shap_table["Combined change"].abs() > 0.001].copy()
        if relevant.empty:
            st.info("No feature attribution change is large enough to display at this precision.")
        else:
            relevant["Feature"] = relevant["Feature"].map(NAMES)
            st.dataframe(relevant.round(3).rename(columns={
                "Heating change": "Heating SHAP change (kWh/m²)",
                "Cooling change": "Cooling SHAP change (kWh/m²)",
                "Combined change": "Combined SHAP change (kWh/m²)",
            }), hide_index=True, use_container_width=True)
            st.caption("Negative SHAP changes correspond to lower model-predicted loads. These are explanations of the model, not measured physical savings.")
    except Exception as exc:
        st.warning("The prediction is available, but its SHAP explanation could not be calculated.")
        with st.expander("Technical details"):
            st.code(str(exc))

    with st.expander("Compare with matching UCI simulation entries (when available)"):
        reference = match_reference(X, y, building, chosen)
        if reference is None:
            st.write("This exact pair of designs does not appear in the UCI dataset.")
        else:
            st.write(f"Change in combined reference loads: {reference:+.2f}% reduction.")
            st.caption("This is a comparison of simulated dataset rows, not independently measured building performance.")

st.divider()
with st.expander("About this research application"):
    st.write("""The model is XGBoost (300 trees, maximum depth 1, learning rate 0.05), trained on
    all 768 configurations of the UCI Energy Efficiency dataset for this demonstration. One editable
    characteristic is changed at a time, while the five structural features remain fixed.
    The full research notebook contains the separate cross-validation and model comparison experiments.
    The dataset contains simulated heating and cooling loads; outputs are research estimates, not metered energy savings.""")
st.caption("UCI Energy Efficiency dataset · Research prototype · Predictions are not engineering design advice")
