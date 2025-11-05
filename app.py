import streamlit as st
import pandas as pd
import joblib
import os
import requests
from dotenv import load_dotenv

# ---------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------
# PAGE CONFIGURATION
# ---------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ---------------------------
# LOAD MODEL AND ENCODERS
# ---------------------------
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("churn_model.pkl")
    encoders = joblib.load("encoder.pkl")  # Can be dict or single LabelEncoder
    return model, encoders

model, encoders = load_model_and_encoders()

# ---------------------------
# LOAD DATASET STRUCTURE
# ---------------------------
df = pd.read_csv("telco_churn.csv")
for col in ["customerID", "Churn"]:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# ---------------------------
# PAGE HEADER
# ---------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
Use this interactive dashboard to predict whether a customer is likely to churn  
and get an **AI-powered explanation** with actionable business recommendations.
""")

# ---------------------------
# SIDEBAR INPUTS
# ---------------------------
st.sidebar.header("🧍‍♂️ Enter Customer Details")

user_input = {}
for col in df.columns:
    if df[col].dtype == "object":
        unique_vals = df[col].unique().tolist()
        user_input[col] = st.sidebar.selectbox(col, unique_vals)
    else:
        user_input[col] = st.sidebar.number_input(
            col, float(df[col].min()), float(df[col].max()), float(df[col].mean())
        )

input_df = pd.DataFrame([user_input])

# ---------------------------
# ENCODE INPUT USING TRAINED ENCODERS
# ---------------------------
encoded_input = input_df.copy()
if isinstance(encoders, dict):
    for col in encoded_input.select_dtypes(include="object").columns:
        if col in encoders:
            try:
                encoded_input[col] = encoders[col].transform(encoded_input[col])
            except ValueError:
                encoded_input[col] = 0
        else:
            encoded_input[col] = 0
else:
    for col in encoded_input.select_dtypes(include="object").columns:
        try:
            encoded_input[col] = encoders.transform(encoded_input[col])
        except Exception:
            encoded_input[col] = 0

encoded_input = encoded_input[df.columns]

# ---------------------------
# FUNCTION: CALL OPENROUTER FOR AI EXPLANATION
# ---------------------------
def get_ai_explanation(summary_text):
    if not api_key or len(api_key) < 10:
        return "⚠️ OpenRouter API key missing or invalid. Please update your .env file."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "openai/gpt-3.5-turbo", # Change model if you want
        "messages": [
            {"role": "system", "content": "You are a senior data analyst explaining churn predictions clearly for business stakeholders."},
            {"role": "user", "content": summary_text}
        ],
        "max_tokens": 250,
        "temperature": 0.6
    }
    response = requests.post(openrouter_url, headers=headers, json=body)
    if response.status_code == 200:
        resp_json = response.json()
        return resp_json["choices"][0]["message"]["content"].strip()
    else:
        return f"AI explanation request failed: {response.status_code} - {response.text}"

# ---------------------------
# PREDICTION & AI EXPLANATION
# ---------------------------
if st.button("🔍 Predict Churn"):
    # ---- Model Prediction ----
    prediction = model.predict(encoded_input)[0]
    proba = model.predict_proba(encoded_input)[0][1]

    # Layout
    left_col, right_col = st.columns([1, 1.2], gap="large")

    # LEFT COLUMN — PREDICTION
    with left_col:
        st.markdown("### 🧠 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ Customer is **likely to churn**.\n\n**Probability:** {proba:.2f}")
        else:
            st.success(f"✅ Customer is **not likely to churn**.\n\n**Probability:** {proba:.2f}")

        st.markdown("---")
        st.markdown("**Input Summary:**")
        st.dataframe(input_df.T, use_container_width=True)

    # RIGHT COLUMN — AI EXPLANATION (OpenRouter)
    with right_col:
        st.markdown("### 🤖 AI-Powered Explanation")

        # You can use model feature importances to build a summary prompt if you want!
        # For now, let's just send input and prediction info:
        factors = ', '.join([f"{col}: {user_input[col]}" for col in df.columns])
        summary_text = f"""
        The model predicted this customer is {'likely' if prediction == 1 else 'unlikely'} to churn.
        Relevant customer details: {factors}.
        Write a 3–5 sentence business explanation and a short actionable recommendation.
        """

        with st.spinner("🤖 Generating AI explanation..."):
            ai_explanation = get_ai_explanation(summary_text)
            st.success(ai_explanation)

