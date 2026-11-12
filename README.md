# Diabetes Prediction using Decision Tree (Streamlit)

A simple Streamlit web application that trains a Decision Tree classifier on
`diabetes.csv`, reports test-set metrics, displays the tree, and predicts a
class for user-entered measurements.

## Files

- `app.py` — Streamlit application
- `diabetes.csv` — dataset used by the app
- `requirements.txt` — Python dependencies

## Run locally

1. Install Python 3.9 or newer.
2. Open a terminal in this project folder.
3. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Deploy using GitHub and Streamlit Community Cloud

1. Sign in to GitHub and create a **new public repository**, for example
   `diabetes-decision-tree-streamlit`.
2. Upload `app.py`, `diabetes.csv`, and `requirements.txt` to the repository
   root (or push this folder using Git).
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose your repository and branch (usually `main`).
5. Set the main file path to `app.py`, then select **Deploy**.
6. Streamlit Community Cloud will install `requirements.txt` and publish the app.
   The deployment page will provide the public app URL.

## Dataset columns

The CSV must contain these columns:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age, Outcome`

`Outcome`: 0 = no diabetes; 1 = diabetes.

## Important note

This is an educational machine-learning demonstration, not a medical device.
Predictions must not be used for diagnosis or treatment decisions.
