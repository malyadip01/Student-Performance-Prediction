# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title='Student Performance Prediction', page_icon='🎓')

st.title("🎓 Student Performance Prediction (xAPI-Edu-Data)")
st.write("Fill the student details and press Predict. The app uses the saved RandomForest model and the same encodings used during training.")

# --- Load saved artifacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')   # dict of LabelEncoders
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

# --- Define the feature list and types ---
# IMPORTANT: match the exact column order used when training the model.
# Replace this list with the exact columns you used in training (order matters).
feature_columns = [
    # Example column order used in the notebook. Adjust if your notebook used a different order.
    'gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
    'SectionID', 'Topic', 'Semester', 'Relation', 'raisedhands',
    'VisITedResources', 'AnnouncementsView', 'ParentAnsweringSurvey',
    'ParentschoolSatisfaction', 'StudentAbsenceDays'
]

# Automatically detect which are categorical (present in encoders) and which are numeric
categorical_cols = [c for c in feature_columns if c in encoders and c != 'Class']
numeric_cols = [c for c in feature_columns if c not in categorical_cols]

st.markdown("---")
st.header("Input student features")

# Collect inputs
input_data = {}

# Categorical: present dropdowns using encoder.classes_
for col in categorical_cols:
    classes = list(encoders[col].classes_)
    # show a friendly label
    display_label = col.replace('_', ' ').title()
    selected = st.selectbox(display_label, classes, key=f"cat_{col}")
    input_data[col] = selected

# Numeric: sliders / number inputs — choose sensible ranges
for col in numeric_cols:
    display_label = col.replace('_', ' ').title()
    # Use heuristics for ranges; user can edit these
    if col.lower() in ['raisedhands', 'visit edresources'.replace(' ', '').lower(), 'visitedresources', 'vis itedresources', 'announcementsview', 'discussion']:
        # Most such fields are 0-100
        val = st.slider(display_label, 0, 100, 10, key=f"num_{col}")
    elif 'absence' in col.lower():
        val = st.number_input(display_label + " (days)", 0, 365, 3, key=f"num_{col}")
    else:
        # default numeric input
        val = st.number_input(display_label, value=0, key=f"num_{col}")
    input_data[col] = val

st.markdown("---")
if st.button("Predict"):
    # Build DataFrame with the correct column order
    df_input = pd.DataFrame([input_data], columns=feature_columns)

    # Encode categorical columns using saved LabelEncoders
    for col in categorical_cols:
        le = encoders[col]
        # If user selection not known to encoder (unlikely if using encoder.classes_), handle gracefully
        try:
            df_input[col] = le.transform(df_input[col].astype(str))
        except Exception as e:
            st.error(f"Value '{df_input[col].iloc[0]}' not in encoder for column {col}.")
            st.stop()

    # Scale numeric features (and all features if scaler expects full vector)
    X_scaled = scaler.transform(df_input.values)

    # Predict
    pred = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
    else:
        proba = None

    # Map predicted numeric label back to original class name using target encoder
    target_le = encoders.get('Class', None)
    if target_le is not None:
        pred_label = target_le.inverse_transform(pred)[0]
    else:
        pred_label = str(pred[0])

    st.success(f"Predicted performance: **{pred_label}**")
    if proba is not None:
        # Show probability for each class using target encoder order
        class_names = target_le.classes_ if target_le is not None else [str(i) for i in range(proba.shape[1])]
        st.write("Class probabilities:")
        prob_df = pd.DataFrame(proba, columns=class_names).T
        prob_df.columns = ['Probability']
        st.table(prob_df)

