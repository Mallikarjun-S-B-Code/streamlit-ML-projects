import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score
)

st.set_page_config(
    page_title="Diabetes Prediction | Decision Tree",
    page_icon="🩺",
    layout="wide"
)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

@st.cache_data
def load_default_data():
    return pd.read_csv("diabetes.csv")

@st.cache_resource
def train_model(data):
    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    # In this dataset, zero values in these measurements are treated as missing.
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, float("nan"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return model, X_test, y_test, predictions, probabilities

st.title("🩺 Diabetes Prediction Using a Decision Tree")
st.write(
    "An educational machine-learning application that classifies records "
    "as **No diabetes (0)** or **Diabetes (1)** using a Decision Tree."
)

with st.sidebar:
    st.header("Project information")
    st.write("**Algorithm:** Decision Tree Classifier")
    st.write("**Dataset:** Pima Indians Diabetes dataset")
    st.write("**Target:** Outcome")
    st.caption("0 = No diabetes | 1 = Diabetes")
    st.divider()
    st.info(
        "This application is for educational demonstration only. "
        "It is not a medical diagnostic tool."
    )

uploaded = st.file_uploader("Optional: upload a CSV dataset", type=["csv"])
try:
    data = pd.read_csv(uploaded) if uploaded is not None else load_default_data()
    missing_cols = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing_cols:
        st.error("CSV is missing required columns: " + ", ".join(missing_cols))
        st.stop()
    data = data[FEATURES + [TARGET]].dropna(subset=[TARGET])
    data[TARGET] = data[TARGET].astype(int)
    if not set(data[TARGET].unique()).issubset({0, 1}):
        st.error("Outcome must contain only 0 and 1.")
        st.stop()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

st.subheader("Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total records", f"{len(data):,}")
m2.metric("Input features", len(FEATURES))
m3.metric("No diabetes", int((data[TARGET] == 0).sum()))
m4.metric("Diabetes", int((data[TARGET] == 1).sum()))

with st.expander("Preview dataset"):
    st.dataframe(data.head(20), use_container_width=True)

try:
    model, X_test, y_test, predictions, probabilities = train_model(data)
except Exception as exc:
    st.error(f"Model training failed. Check that the CSV has enough examples per class. Details: {exc}")
    st.stop()

st.subheader("Model performance")
accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else float("nan")
c1, c2, c3 = st.columns(3)
c1.metric("Test accuracy", f"{accuracy:.1%}")
c2.metric("ROC-AUC", f"{auc:.3f}" if pd.notna(auc) else "N/A")
c3.metric("Test records", len(y_test))

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    st.dataframe(
        pd.DataFrame(cm, index=["Actual: No", "Actual: Yes"],
                     columns=["Predicted: No", "Predicted: Yes"]),
        use_container_width=True
    )
with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y_test, predictions, target_names=["No diabetes", "Diabetes"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.subheader("Decision tree visualization")
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(
    model.named_steps["tree"],
    feature_names=FEATURES,
    class_names=["No diabetes", "Diabetes"],
    filled=True, rounded=True, impurity=False, fontsize=8, ax=ax
)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Predict for a new record")
st.write("Enter the measurements below and select **Predict**.")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
        bp = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
        skin = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=25.0)
    with col2:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=28.5)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.45)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    patient = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }], columns=FEATURES)
    predicted_class = int(model.predict(patient)[0])
    risk_probability = float(model.predict_proba(patient)[0, 1])
    if predicted_class == 1:
        st.warning(f"Model result: Diabetes class (1). Model probability: {risk_probability:.1%}")
    else:
        st.success(f"Model result: No-diabetes class (0). Model probability for diabetes: {risk_probability:.1%}")
    st.caption(
        "The probability is produced by this model and is not a clinically validated "
        "estimate of an individual's medical risk. Consult a qualified healthcare professional."
    )

st.subheader("Download test predictions")
results = X_test.copy()
results["Actual_Outcome"] = y_test.values
results["Predicted_Outcome"] = predictions
results["Diabetes_Class_Probability"] = probabilities
st.download_button(
    "Download predictions as CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="diabetes_test_predictions.csv",
    mime="text/csv"
)

st.caption("Academic project • Decision Tree Classification • Python + Streamlit")
