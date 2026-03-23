import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image
import io

from src.rag_pipeline import get_rag_chain
from CV_model_building.preprocess import preprocess_image_inference

# 🔥 Load model
model = tf.keras.models.load_model("CV_model_building/xray_model.keras")

# 🔥 Load class names
with open("CV_model_building/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

app = FastAPI()


# ✅ Pydantic model (USED properly)
class PatientData(BaseModel):
    age: int
    oxygen: int
    fever: str
    breathlessness: str
    cough_days: int
    cough_type: str
    chest_pain: bool
    night_sweats: bool
    weight_loss: bool
    smoking: bool
    comorbidity: bool


# 🔥 Risk Score
def calculate_risk_score(pred_disease, confidence, data: PatientData):

    score = 0
    score += confidence * 40

    if data.age > 60:
        score += 15
    elif data.age > 40:
        score += 8

    if data.oxygen < 90:
        score += 30
    elif data.oxygen < 95:
        score += 15

    if data.fever == "high":
        score += 10

    if data.breathlessness == "severe":
        score += 15
    elif data.breathlessness == "mild":
        score += 5

    if data.cough_days > 14:
        score += 10

    if data.cough_type == "blood":
        score += 20

    if data.night_sweats:
        score += 10

    if data.weight_loss:
        score += 10

    if data.smoking:
        score += 5

    if data.comorbidity:
        score += 10

    return score


def get_risk_level(score):
    if score >= 80:
        return "High"
    elif score >= 50:
        return "Moderate"
    return "Low"


# 🚀 MAIN API
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),

    # ✅ Proper form inputs (NO JSON string mess)
    age: int = Form(...),
    oxygen: int = Form(...),
    fever: str = Form(...),
    breathlessness: str = Form(...),
    cough_days: int = Form(...),
    cough_type: str = Form(...),
    chest_pain: bool = Form(...),
    night_sweats: bool = Form(...),
    weight_loss: bool = Form(...),
    smoking: bool = Form(...),
    comorbidity: bool = Form(...)
):

    # 🔹 Convert to Pydantic object
    patient = PatientData(
        age=age,
        oxygen=oxygen,
        fever=fever,
        breathlessness=breathlessness,
        cough_days=cough_days,
        cough_type=cough_type,
        chest_pain=chest_pain,
        night_sweats=night_sweats,
        weight_loss=weight_loss,
        smoking=smoking,
        comorbidity=comorbidity
    )

    # 🔹 Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # 🔹 Preprocess
    img = preprocess_image_inference(image)

    # 🔹 Predict
    preds = model.predict(img)[0]
    idx = np.argmax(preds)

    disease = class_names[idx]
    confidence = float(preds[idx])
    
    result_dict = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

    # 🔹 Risk
    risk_score = calculate_risk_score(disease, confidence, patient)
    risk_level = get_risk_level(risk_score)

    # 🔥 Build RAG Query (IMPORTANT)
    query = f"""
You are an advanced AI medical assistant generating a structured clinical-style report for chest X-ray analysis.

==================== AI RADIOLOGY REPORT ====================

🔍 Model Prediction Summary:
- Primary Prediction: {disease}
- Confidence Score: {confidence:.2f}
- Risk Classification: {risk_level}

📊 Probability Distribution (All Conditions):
{result_dict}

👤 Patient Profile:
- Age: {patient.age}
- Oxygen Saturation: {patient.oxygen}%

🩺 Reported Symptoms:
- Fever: {patient.fever}
- Breathlessness: {patient.breathlessness}
- Cough: {patient.cough_days} days ({patient.cough_type})
- Chest Pain: {patient.chest_pain}
- Night Sweats: {patient.night_sweats}
- Weight Loss: {patient.weight_loss}

==============================================================

📌 Instructions for Analysis:

- Carefully analyze ALL probabilities, not just the highest.
- If TB or Pneumonia probability > 0.2, consider early-stage possibility.
- Correlate symptoms with model predictions.
- If symptoms strongly indicate a disease, highlight concern even if model confidence is moderate.
- Maintain a cautious, clinical tone.
- Do NOT give a definitive diagnosis.

==============================================================

🧾 Generate the report in the following structured format:

1. 🧠 Summary of Findings:
   - Brief overview of the X-ray interpretation

2. 🔬 Clinical Interpretation:
   - Explain possible conditions (Normal / TB / Pneumonia)
   - Mention uncertainty if present

3. ⚠️ Risk Assessment:
   - Clearly state if condition appears Mild / Moderate / Concerning
   - Highlight any red flags

4. 🩺 Symptom Correlation:
   - Connect symptoms with possible diseases

5. 📋 Recommended Next Steps:
   - Suggest tests (e.g., sputum test, CT scan, blood test)
   - Recommend consulting a doctor

6. ⚠️ Medical Disclaimer:
   - State this is an AI-assisted analysis and not a confirmed diagnosis

==============================================================

Ensure the tone is:
- Clear
- Professional
- Reassuring but cautious
- Easy for a non-medical person to understand
"""
    # 🔥 RAG CHAIN (Metadata filter is used INSIDE this)
    chain = get_rag_chain(disease, confidence)

    response = chain.invoke({"question":query})

    return {
        "disease": disease,
        "confidence": round(confidence, 3),
        "risk_score": int(risk_score),
        "risk_level": risk_level,
        "response": response["answer"]
    }
    
# 🔥 GLOBAL CHAT CHAIN (for memory)
chat_chain = None

@app.post("/chat")
async def chat(
    question: str = Form(...),
    disease: str = Form(...),
    confidence: float = Form(...)
):
    global chat_chain

    # Create once → keep memory
    if chat_chain is None:
        chat_chain = get_rag_chain(disease, confidence)

    response = chat_chain.invoke({"question": question})

    return {
        "answer": response["answer"]
    }