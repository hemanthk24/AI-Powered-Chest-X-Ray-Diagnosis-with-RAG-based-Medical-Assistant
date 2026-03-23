import streamlit as st
import requests

st.set_page_config(page_title="AI Chest X-ray Assistant", layout="centered")

st.markdown(
    "<h1 style='text-align: center;'>🩺 AI Chest X-ray Assistant</h1>",
    unsafe_allow_html=True
)

# 🔥 SESSION STATE
if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# 📸 Upload image
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded X-ray", width="content")

st.subheader("Patient Details")

# 🧍 Basic Info
age = st.number_input("Age", 1, 100, 30)
oxygen = st.slider("Oxygen Level (SpO2)", 70, 100, 95)

# 🤒 Symptoms
fever = st.selectbox("Fever", ["none", "low", "high"])
breathlessness = st.selectbox("Breathlessness", ["none", "mild", "severe"])

cough_days = st.number_input("Cough Duration (days)", 0, 60, 5)
cough_type = st.selectbox("Cough Type", ["dry", "mucus", "blood"])

chest_pain = st.checkbox("Chest Pain")
night_sweats = st.checkbox("Night Sweats")
weight_loss = st.checkbox("Weight Loss")

# ⚠️ Risk Factors
smoking = st.checkbox("Smoking")
comorbidity = st.checkbox("Comorbidity (Diabetes/Asthma)")


# 🔥 Analyze Button
if st.button("Analyze"):

    if uploaded_file is None:
        st.warning("Please upload an X-ray image")
    else:
        with st.spinner("Analyzing..."):

            files = {
                "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
            }

            data = {
                "age": age,
                "oxygen": oxygen,
                "fever": fever,
                "breathlessness": breathlessness,
                "cough_days": cough_days,
                "cough_type": cough_type,
                "chest_pain": str(chest_pain).lower(),
                "night_sweats": str(night_sweats).lower(),
                "weight_loss": str(weight_loss).lower(),
                "smoking": str(smoking).lower(),
                "comorbidity": str(comorbidity).lower()
            }

            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files,
                    data=data,
                    timeout=120
                )

                if response.status_code == 200:
                    st.session_state.result = response.json()
                    st.session_state.chat_history = []  # reset chat on new analysis
                else:
                    st.error(f"API Error: {response.text}")

            except Exception as e:
                st.error(f"Error connecting to API: {e}")


# 🔥 SHOW RESULT (persistent)
if st.session_state.result:

    result = st.session_state.result

    # 🧠 Prediction
    st.subheader("🧠 Prediction")
    st.success(f"{result['disease']} ({result['confidence']:.2f})")

    # ⚠️ Risk
    st.subheader("⚠️ Risk Level")
    if result["risk_level"] == "High":
        st.error(f"{result['risk_level']} (Score: {result['risk_score']})")
    elif result["risk_level"] == "Moderate":
        st.warning(f"{result['risk_level']} (Score: {result['risk_score']})")
    else:
        st.info(f"{result['risk_level']} (Score: {result['risk_score']})")

    # 📄 Report
    st.subheader("📄 AI Medical Report")
    st.write(result["response"])

    st.divider()

    # 💬 CHAT SECTION
    st.subheader("💬 Chat with AI Doctor")

    user_input = st.chat_input("Ask about your condition...")

    if user_input:

        # Save user message
        st.session_state.chat_history.append(("user", user_input))

        with st.spinner("Thinking..."):

            data = {
                "question": user_input,
                "disease": result["disease"],
                "confidence": result["confidence"]
            }

            try:
                response = requests.post(
                    "http://127.0.0.1:8000/chat",
                    data=data
                )

                if response.status_code == 200:
                    answer = response.json()["answer"]
                else:
                    answer = "Error getting response from server."

            except Exception as e:
                answer = f"Connection error: {e}"

        # Save AI response
        st.session_state.chat_history.append(("assistant", answer))


    # 🔥 DISPLAY CHAT HISTORY
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)