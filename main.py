import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image
import io

from src.rag_pipeline import get_rag_chain
from CV_model_building.preprocess import preprocess_image_inference

# definig for model
def sparse_focal_loss(gamma=2.0):
    def loss(y_true, y_pred):

        # 🔥 FIX SHAPE
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, [-1])   # ✅ important fix

        # 🔥 cross entropy
        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred
        )

        # 🔥 focal part
        p_t = tf.exp(-ce)
        focal_loss = (1 - p_t) ** gamma * ce

        return focal_loss

    return loss

# 🔥 Load model
model = tf.keras.models.load_model("CV_model_building/chest_xray_model.keras",
                                   custom_objects={"loss": sparse_focal_loss(gamma=2.0)})

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
You are an advanced AI medical assistant generating a structured, clinical-style chest X-ray report.

                 AI RADIOLOGY REPORT

🔍 Model Prediction Summary:
- Primary Prediction: {disease}
- Confidence Score: {confidence:.2f}
- Risk Classification: {risk_level}

📊 Probability Distribution:
{result_dict}

👤 Patient Profile:
- Age: {patient.age}
- Oxygen Saturation: {patient.oxygen}%

🩺 Symptoms:
- Fever: {patient.fever}
- Breathlessness: {patient.breathlessness}
- Cough: {patient.cough_days} days ({patient.cough_type})
- Chest Pain: {patient.chest_pain}
- Night Sweats: {patient.night_sweats}
- Weight Loss: {patient.weight_loss}

👤 Habits & Comorbidities:
- Smoking: {patient.smoking}
- Comorbidity: {patient.comorbidity}

📌 CLINICAL ANALYSIS GUIDELINES:

1. DO NOT rely only on the top prediction.
2. Carefully analyze ALL probabilities.
3. If second-highest probability is close to the first → highlight uncertainty.
4. If TB or Pneumonia probability ≥ 0.20 → consider early-stage possibility.
5. If model confidence is LOW (<0.4):
   → Treat prediction as "Uncertain / Possible Conditions"
   → Do NOT strongly conclude any disease.
6. If prediction is "Unknown":
   → Clearly state model uncertainty
   → Suggest further diagnostic testing
7. Strongly correlate symptoms with predictions:
   - If symptoms strongly indicate a disease, highlight it EVEN if model confidence is moderate.
8. Avoid definitive diagnosis — maintain clinical caution.

==============================================================

🧾 REPORT FORMAT (STRICT):

1. 🧠 Summary of Findings:
   - Mention predicted condition OR uncertainty
   - Highlight if multiple conditions are possible

2. 🔬 Clinical Interpretation:
   - Explain likelihood of:
     • Normal
     • Pneumonia
     • Tuberculosis
   - Clearly mention if case is:
     → Confident / Moderate / Uncertain

3. ⚠️ Risk Assessment:
   - Classify as: Low / Moderate / Concerning
   - Highlight red flags (low oxygen, long cough, blood cough, weight loss)

4. 🩺 Symptom Correlation:
   - Connect symptoms with possible conditions
   - Highlight mismatches if any

5. 📋 Recommended Next Steps:
   - Suggest appropriate medical tests:
     • Chest CT scan
     • Sputum test (for TB)
     • Blood tests
     • Doctor consultation
   - If uncertain → emphasize need for further evaluation

6. ⚠️ Medical Disclaimer:
   - Clearly state:
     "This is an AI-assisted analysis and not a confirmed medical diagnosis."

==============================================================

🧠 IMPORTANT BEHAVIOR RULES:

- Be cautious, not alarming
- If disease is "UNKNOWN" tell that image entered may not be crct or not a chest x-ray
- Be informative, not vague
- Be clear for non-medical users
- If uncertain → say it clearly (DO NOT GUESS)
- Prioritize patient safety over confidence

==============================================================

Generate a clear, structured, and professional medical-style report.
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