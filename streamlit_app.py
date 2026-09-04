import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
from datetime import datetime

# --- EDIT YOUR TEAM DETAILS HERE ---
TEAM_NAME = "Your Team Name"
SUBMISSION_DATE = "March 04, 2026"
PROJECT_VERSION = "v1.0"

# --- PAGE CONFIGURATION & MEDICAL STYLING ---
st.set_page_config(
    page_title="Diabetic Retinopathy Clinical Portal", 
    page_icon="🩺", 
    layout="wide"
)

# Custom Clinical CSS Theme
st.markdown("""
    <style>
    /* Global Page Styling */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-title { font-size: 2.2rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1rem; color: #93C5FD; margin-bottom: 0.8rem; }
    .team-meta { font-size: 0.85rem; color: #E2E8F0; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 8px; }
    
    /* Medical Card Enclosures */
    .med-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #2563EB;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# Header Banner with Team Metadata
st.markdown(f"""
    <div class="header-box">
        <div class="main-title">🩺 Clinical Decision Support Portal</div>
        <div class="subtitle">Automated Diagnostic Assessment & Microvascular Evaluation | Ophthalmology AI Support</div>
        <div class="team-meta">
            <strong>Developed by:</strong> {TEAM_NAME} &nbsp;|&nbsp; 
            <strong>Submission Date:</strong> {SUBMISSION_DATE} &nbsp;|&nbsp; 
            <strong>Version:</strong> {PROJECT_VERSION}
        </div>
    </div>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE FOR INPUT TRACKING ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Filename', 'Predicted Finding', 'Confidence', 'Timestamp'])

# --- AUTOMATIC MODEL DOWNLOAD ---
MODEL_FILE_ID = '1liKVBcah0zt-Yku3wIKJ20_idwwcEmh0' 
MODEL_PATH = "diabetic_retinopathy_resnet18.pth"

@st.cache_resource
def load_medical_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_medical_model()
except Exception as e:
    st.error("⚠️ Model Loading Error: Unable to fetch model weights from Google Drive.")

# --- MEDICAL TRANSFORMS & CLINICAL GUIDELINES ---
predict_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_medical_guidelines(diagnosis_class):
    if "Diseased" in diagnosis_class:
        return (
            "⚠️ CLINICAL STATUS: ACTION REQUIRED\n\n"
            "💊 Standard Management Options:\n"
            "• Systemic Control: Blood glucose regulation (e.g., Metformin, Insulin therapy), "
            "blood pressure control via ACE inhibitors, and lipid-lowering agents.\n"
            "• Ocular Therapy: Evaluation for intravitreal Anti-VEGF injections (e.g., Ranibizumab, Aflibercept) "
            "or corticosteroid implants.\n"
            "• Surgical Options: Advanced proliferative stages may require laser photocoagulation or vitrectomy.\n\n"
            "🚨 Disclaimer: Generated for screening support. Consult a licensed ophthalmologist for treatment planning."
        )
    else:
        return (
            "✅ CLINICAL STATUS: ROUTINE FOLLOW-UP\n\n"
            "🩺 Preventative Care Guidelines:\n"
            "• Screening: Schedule annual comprehensive dilated eye examinations.\n"
            "• Maintenance: Continue monitoring HbA1c, blood pressure, and lipid levels.\n"
            "• Lifestyle Support: Maintain a balanced nutritional plan and regular cardiovascular exercise.\n\n"
            "⚠️ Disclaimer: General evaluation only. Regular physical eye exams remain necessary."
        )

# --- SIDEBAR CLINICAL NAVIGATION ---
st.sidebar.image("https://img.icons8.com/color/96/ophthalmology.png", width=70)
st.sidebar.title("Clinical Navigation")

page = st.sidebar.radio("Select View:", [
    "🩻 Diagnostic Image Screening", 
    "📋 Patient Assessment Logs"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Team:** {TEAM_NAME}")
st.sidebar.markdown(f"**Date:** {SUBMISSION_DATE}")

# PAGE 1: DIAGNOSTIC SCREENING
if page == "🩻 Diagnostic Image Screening":
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("1. Fundus Image Upload")
    uploaded_file = st.file_uploader("Upload Retinal Scan (JPG, PNG)", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Dual Column View
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown('<div class="med-card">', unsafe_allow_html=True)
            st.subheader("Retinal Imaging View")
            st.image(image, use_container_width=True, caption=uploaded_file.name)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction
        img_t = predict_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)[0]
            _, predicted = outputs.max(1)
            
        classes = ['Diseased (Diabetic Retinopathy)', 'Normal (Healthy)']
        diagnosis = classes[predicted.item()]
        confidence = float(probs[predicted.item()]) * 100
        
        # Auto-log to history state if not already logged
        if not ((st.session_state.history['Filename'] == uploaded_file.name).any()):
            new_entry = pd.DataFrame([{
                'Filename': uploaded_file.name,
                'Predicted Finding': diagnosis,
                'Confidence': f"{confidence:.1f}%",
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)
        
        with col2:
            st.markdown('<div class="med-card">', unsafe_allow_html=True)
            st.subheader("Automated Analysis Output")
            
            st.progress(float(probs[0]), text=f"Diabetic Retinopathy: {probs[0]*100:.1f}%")
            st.progress(float(probs[1]), text=f"Healthy Retina: {probs[1]*100:.1f}%")
            
            if "Diseased" in diagnosis:
                st.error(f"Diagnostic Finding: {diagnosis}")
            else:
                st.success(f"Diagnostic Finding: {diagnosis}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Management Guidelines
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.subheader("2. Clinical Guidelines")
        st.text_area("Protocol Recommendations", value=get_medical_guidelines(diagnosis), height=200)
        st.markdown('</div>', unsafe_allow_html=True)

# PAGE 2: PATIENT RECORDS LOG
elif page == "📋 Patient Assessment Logs":
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("Historic Upload Logs")
    if len(st.session_state.history) > 0:
        st.dataframe(st.session_state.history, use_container_width=True)
    else:
        st.info("No saved records found.")
    st.markdown('</div>', unsafe_allow_html=True)
