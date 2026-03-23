# 🩺 MediAssist AI (Chest X-ray Diagnostic Assistant with RAG-based Clinical Explanation)

## Overview

**MediAssist AI** is an end-to-end AI-powered healthcare assistant designed to analyze chest X-ray images, predict potential lung diseases, assess patient risk, and provide explainable clinical insights using Retrieval-Augmented Generation (RAG).

This system integrates **Computer Vision + Machine Learning + Generative AI + FastAPI + Streamlit** to deliver a complete intelligent diagnostic workflow.

---

## Key Features

* **Chest X-ray Classification**

  * Detects conditions such as:

    * Normal
    * Pneumonia
    * Tuberculosis
  * Built using deep learning (CNN)

* **High Accuracy Model**

  * Achieved **92.77% validation accuracy**

* **Risk Assessment Engine**

  * Combines:

    * Model confidence
    * Patient vitals (SpO2, age)
    * Symptoms (fever, cough, etc.)
  * Outputs risk levels: **Low / Moderate / High**

* 📄 **AI-Generated Clinical Report**

  * Structured medical-style explanation
  * Includes:

    * Interpretation
    * Symptom correlation
    * Recommendations

* 🤖 **RAG-based Medical Assistant**

  * Uses vector database + LLM
  * Provides **context-aware explanations**
  * Filters knowledge based on predicted disease

* 💬 **Interactive Chat System**

  * Users can ask follow-up questions
  * Maintains conversation context

---

## 🏗️ System Architecture

```
Streamlit (Frontend UI)
        ↓
FastAPI (Backend API)
        ↓
CNN Model (Image Classification)
        ↓
Risk Scoring Engine
        ↓
RAG Pipeline (Retriever + LLM)
        ↓
Response → UI (Report + Chat)
```

---

## 🧪 Tech Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI
* **Machine Learning:** TensorFlow / Keras
* **Data Processing:** NumPy, Pandas
* **RAG Pipeline:** LangChain
* **Embeddings:** HuggingFace (MiniLM)
* **Vector Database:** Pinecone
* **LLM:** OpenAI (GPT-4o-mini)
* **Other:** PIL, Requests

---

## ⚙️ How It Works

1. User uploads a chest X-ray image via Streamlit
2. Image is sent to FastAPI using multipart/form-data
3. Model processes image and predicts disease
4. Risk score is calculated using patient inputs
5. RAG pipeline retrieves relevant medical context
6. LLM generates structured clinical explanation
7. Results are displayed along with chat interface

---

## 📂 Project Structure

```
├── main.py                  # FastAPI backend
├── app.py                   # Streamlit frontend
├── src/
│   ├── rag_pipeline.py
│   ├── vector_store.py
│   ├── create_retriever.py
│   ├── docs_loader.py
│   └── prompt.py
├── CV_model_building/
│   ├── xray_model.keras
│   ├── preprocess.py
│   └── class_names.pkl
├── data/
│   ├── Pneumonia.pdf
│   └── Tuberculosis.pdf
```

---

## ▶️ Running the Project

### 1️⃣ Clone the repository

```bash
git clone <your-repo-link>
cd project-folder
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run FastAPI backend

```bash
uvicorn main:app --reload
```

### 5️⃣ Run Streamlit frontend

```bash
streamlit run app.py
```

---

## 📊 Model Performance

* Validation Accuracy: **92.77%**
* Model Type: CNN (Deep Learning)
* Input: Chest X-ray images
* Output: Multi-class classification

---

## 🔍 Future Improvements

* Grad-CAM visualization for explainability
* Database integration for patient history
* Deployment (AWS / Render)
* Real-time streaming responses
* Multi-disease support

---

## ⚠️ Disclaimer

This system is intended for **educational and research purposes only**.
It should **not be used as a substitute for professional medical diagnosis**.

---

## 👨‍💻 Author

**Hemanth Goud**
AI & Data Science Enthusiast

---
****
