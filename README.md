# Energy-Efficient Building Design

Course project by Shabib Hyder Gazi and Satnam Singh Khokhar, Georgia State University.

This Streamlit app estimates heating and cooling loads for simulated residential building configurations and suggests changes that reduce the model's predicted combined load.

## Run locally

1. Install Python 3.10 or newer.
2. Install requirements: `pip install -r requirements.txt`
3. Start the application: `streamlit run app.py`
4. Use the sidebar to select one of the dataset's 12 structures and specify its orientation, glazing area and glazing distribution.
5. Read the predicted loads, optional design recommendation, and SHAP feature contributions.

The application downloads the UCI Energy Efficiency dataset when first started and trains a depth-1 XGBoost model. It needs internet access for the initial dataset fetch.

## Deploy to Streamlit Community Cloud

Upload `app.py` and `requirements.txt` to the root of a GitHub repository. In [Streamlit Community Cloud](https://share.streamlit.io/), select that repository and choose `app.py` as the main file. Share the resulting app URL as part of the project submission.

## Notes

- Building configurations and target loads come from simulated research data, not metered buildings.
- The dashboard model is trained on the full dataset for interaction. The research notebook contains the separate cross-validation experiments.
- The application searches permitted changes to **one adjustable feature at a time**, not arbitrary whole-building redesigns.
- SHAP explanations describe the model's prediction, not verified physical causation.
- The app requires an internet connection on first launch to fetch the dataset.
