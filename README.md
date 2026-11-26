# Energy-Efficient Building Design Recommender

**Course research project — Shabib Hyder Gazi and Satnam Singh Khokhar**  
Department of Computer Science, Georgia State University

This dashboard predicts heating and cooling loads for a residential building design. It can suggest a change to orientation, glazing area, or glazing distribution and explains how that change affects the model's prediction using SHAP.

## Files

- `app.py` — Streamlit interface
- `model.py` — dataset loading, training, recommendation search, SHAP comparison
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — optional theme settings

The accompanying **Explainable_Method** Colab notebook contains the research experiments, model comparisons, and cross-validation. The app uses the **same final demonstration model settings**: depth-1 XGBoost, 300 estimators, learning rate 0.05, seed 42. It trains on all 768 rows for demonstration only.

## Run locally

1. Install Python 3.10 or newer.
2. From the project folder, run `pip install -r requirements.txt`.
3. Run `streamlit run app.py`.
4. Open the address shown in your terminal.

On first launch, the app downloads the UCI Energy Efficiency dataset (ID 242) and trains two regression estimators. An internet connection is needed for this first dataset request. Streamlit caches the dataset and trained model after loading.

## Deploy with Streamlit Community Cloud

1. Upload **all files**, including `model.py` and `requirements.txt`, to a GitHub repository. Include `.streamlit/config.toml` if desired.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Create an app from the repository's `main` branch, setting **Main file path** to `app.py`.
4. Deploy, then test at least one building example and verify that its results appear.

## Using the app

1. Select one of the 12 simulated structural configurations, or enter five structural characteristics manually.
2. Choose orientation (2–5), glazing area (0, 0.10, 0.25, 0.40), and glazing distribution (0 when glazing is zero; otherwise 1–5).
3. Optionally keep glazing fixed or choose a minimum permissible glazing area. The default improvement threshold is 5%.
4. Read the predicted heating, cooling, and combined loads.
5. If a design meets the threshold, inspect the recommended modification, predicted improvement, and SHAP explanation table. Otherwise the app clearly reports that no qualifying recommendation was found.

### Suggested check

Using the dataset structure that corresponds to **building 668**, set orientation 2, glazing area 0.40, glazing distribution 3, minimum glazing 0.10, and threshold 5%. The exact predictions should match the final Colab notebook demo when using the same inputs and package versions.

## Scope and limitations

The UCI targets are building **simulation outputs**. The app is a course research prototype, not a certified building-energy tool. The app's model is trained using all 768 data rows; the independent prediction/generalization experiments are documented in the Colab notebook. A SHAP explanation describes changes to the model's prediction, not a physical guarantee of energy savings.
