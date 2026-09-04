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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- EDIT YOUR TEAM DETAILS HERE ---
TEAM_NAME = "ML--5th Floor--Group 3"
SUBMISSION_DATE = "Sept 5, 2026"
PROJECT_MODEL = "CNN and Deep Learning"

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

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1rem;
    color: #93C5FD;
    margin-bottom: 0.8rem;
}

.team-meta {
    font-size: 0.85rem;
    color: #E2E8F0;
    border-top: 1px solid rgba(255,255,255,0.2);
    padding-top: 8px;
}

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

/* Severity Color Alert Boxes */
.alert-normal {
    background-color: #D1FAE5;
    color: #065F46;
    border-left: 6px solid #10B981;
    padding: 16px;
    border-radius: 8px;
    font-weight: 600;
    margin-top: 10px;
}

.alert-stage1 {
    background-color: #FEF3C7;
    color: #92400E;
    border-left: 6px solid #F59E0B;
    padding: 16px;
    border-radius: 8px;
    font-weight: 600;
    margin-top: 10px;
}

.alert-stage2 {
    background-color: #FEE2E2;
    color: #991B1B;
    border-left: 6px solid #F87171;
    padding: 16px;
    border-radius: 8px;
    font-weight: 600;
    margin-top: 10px;
}

.alert-stage3 {
    background-color: #7F1D1D;
    color: #FFFFFF;
    border-left: 6px solid #DC2626;
    padding: 16px;
    border-radius: 8px;
    font-weight: 700;
    margin-top: 10px;
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
        <strong>Model Architecture:</strong> {PROJECT_MODEL}
    </div>
</div>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE FOR INPUT TRACKING ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Filename', 'Predicted', 'Ground Truth', 'Confidence', 'Timestamp'])

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

# --- MEDICAL TRANSFORMS & STAGED CLINICAL GUIDELINES ---
predict_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_severity(diseased_prob):
    """
    Evaluates DR staging based on diseased probability score:
    - Normal: diseased_prob < 0.40 (Green)
    - Stage 1 (Mild DR): 0.40 <= diseased_prob < 0.60 (Yellow)
    - Stage 2 (Intermediate DR): 0.60 <= diseased_prob < 0.75 (Soft Red)
    - Stage 3 (Severe DR): diseased_prob >= 0.75 (Bright Red)
    """
    if diseased_prob < 0.40:
        stage_name = "Normal (Healthy Retina)"
        alert_class = "alert-normal"
        guidelines = (
            "✅ CLINICAL STATUS: ROUTINE FOLLOW-UP\n\n"
            "🩺 Preventative Care Guidelines:\n"
            "• Screening: Schedule annual comprehensive dilated eye examinations.\n"
            "• Glycemic Control: Continue routine HbA1c and blood pressure monitoring.\n"
            "• Lifestyle Maintenance: Maintain a balanced dietary plan and regular physical activity.\n\n"
            "⚠️ Disclaimer: General screening output. Regular dilated eye exams remain required."
        )
    elif 0.40 <= diseased_prob < 0.60:
        stage_name = "Stage 1: Mild Diabetic Retinopathy"
        alert_class = "alert-stage1"
        guidelines = (
            "🟡 CLINICAL STATUS: STAGE 1 - MILD DR (MONITORING REQUIRED)\n\n"
            "💊 Non-Pharmacological & Monitoring Guidelines:\n"
            "• Strict Glycemic Control: Optimize blood glucose levels to reduce progression risk (target HbA1c < 7.0%).\n"
            "• Systemic Monitoring: Regulate blood pressure and lipid profile closely.\n"
            "• Re-evaluation: Schedule a follow-up dilated fundus exam within 6 to 12 months.\n"
            "• Patient Education: Instruct patient on recognizing early visual disturbance signs.\n\n"
            "🚨 Disclaimer: Consult a licensed ophthalmologist for clinical evaluation."
        )
    elif 0.60 <= diseased_prob < 0.75:
        stage_name = "Stage 2: Intermediate Diabetic Retinopathy"
        alert_class = "alert-stage2"
        guidelines = (
            "🔴 CLINICAL STATUS: STAGE 2 - INTERMEDIATE DR (CLINICAL INTERVENTION RECOMMENDED)\n\n"
            "🩺 Intermediate Management Protocols:\n"
            "• Specialty Referral: Prompt referral to a retinal specialist for comprehensive evaluation.\n"
            "• Diagnostic Imaging: Consider Optical Coherence Tomography (OCT) to screen for macular edema.\n"
            "• Aggressive Risk Control: Strict BP control (< 130/80 mmHg) and lipid-lowering therapy.\n"
            "• Follow-Up Schedule: Repeat clinical retinal evaluation within 3 to 6 months.\n\n"
            "🚨 Disclaimer: Requires clinical correlation and professional ophthalmic assessment."
        )
    else:  # diseased_prob >= 0.75
        stage_name = "Stage 3: Severe Diabetic Retinopathy"
        alert_class = "alert-stage3"
        guidelines = (
            "🚨 CLINICAL STATUS: STAGE 3 - SEVERE DR (URGENT MEDICAL & PHARMACOLOGICAL ACTION)\n\n"
            "💊 Urgent Pharmacological & Therapeutic Interventions:\n"
            "• Ocular Pharmacotherapy: Evaluation for intravitreal Anti-VEGF therapy (e.g., Ranibizumab, Aflibercept) or intravitreal corticosteroids.\n"
            "• Advanced Procedures: Immediate evaluation for Panretinal Photocoagulation (PRP) laser therapy or surgical vitrectomy if hemorrhages occur.\n"
            "• Urgent Referral: High-priority appointment with an ophthalmologist/retinal surgeon within 1-2 weeks.\n"
            "• Strict Medical Management: Immediate multidisciplinary care with endocrinology for glycemic stabilization.\n\n"
            "🚨 Critical Disclaimer: High risk of vision loss. Immediate specialist management required."
        )
    
    return stage_name, alert_class, guidelines

# --- SIDEBAR CLINICAL NAVIGATION ---
st.sidebar.image("https://img.icons8.com/color/96/ophthalmology.png", width=70)
st.sidebar.title("Clinical Navigation")
page = st.sidebar.radio("Select View:", [
    "📖 Overview & Model Architecture",
    "🩻 Diagnostic Image Screening",
    "📊 Input Metrics & Confusion Matrix",
    "📋 Patient Assessment Logs"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Team:** {TEAM_NAME}")
st.sidebar.markdown(f"**Date:** {SUBMISSION_DATE}")

# PAGE 0: OVERVIEW & COLAB MODEL ARCHITECTURE
if page == "📖 Overview & Model Architecture":
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("💡 Why We Built This Diabetic Retinopathy (DR) AI Screening Project")
    st.markdown("""
    Diabetic Retinopathy (DR) is an eye disease caused by high blood sugar levels damaging the tiny blood vessels in the back of the eye (retina). It is one of the leading causes of preventable blindness worldwide.
    * **The Problem:** Early DR has no symptoms, meaning patients often don't realize they have it until permanent vision loss occurs. Manual eye exams require trained ophthalmologists, who are scarce in many regions.
    * **Our Solution:** This AI tool provides an automated, rapid preliminary screening of retinal scan images (128x128 pixels). It instantly alerts patients and medical professionals whether a scan shows signs of **Diabetic Retinopathy** or a **Healthy Retina**, enabling fast triage and timely medical intervention.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("🧠 Google Colab CNN Model Architecture & Fine-Tuning Setup")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**1. Backbone Architecture: ResNet-18**")
        st.markdown("""
        * **Base Network:** Pre-trained `ResNet-18` (Convolutional Neural Network) utilizing residual skip-connections to retain deep image feature maps.
        * **Fine-Tuning Strategy:** Layers 1 through 3 were frozen (`requires_grad = False`) to preserve general image features. Layer 4 and the custom fully-connected header were unfrozen for targeted medical adaptation.
        * **Custom Fully-Connected Head:**
            * `Linear` layer: 512 input features → 256 nodes
            * `ReLU` activation function for non-linearity
            * `Dropout(0.4)`: 40% neuron dropout to prevent model overfitting
            * `Linear` layer: 256 nodes → 2 output classes (Diseased vs. Normal)
        """)
    with col_b:
        st.markdown("**2. Training Pipeline & Class Imbalance Handling**")
        st.markdown("""
        * **Data Preprocessing & Augmentation:** Images resized to `128x128`, transformed to PyTorch Tensors, normalized (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), and augmented with `RandomHorizontalFlip()` and `RandomRotation(15°)`.
        * **Data Split:** 80% Training / 20% Validation (`random_split`).
        * **Loss Function:** `CrossEntropyLoss` weighted inversely proportional to class frequencies to combat dataset imbalance.
        * **Optimizer:** Per-layer `Adam` optimizer (Layer 4 `lr = 0.00001`, Fully-Connected head `lr = 0.0001`).
        * **Batch Size & Epochs:** `Batch Size = 32`, `Epochs = 5`.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("📈 Google Colab Model Training & Validation Evaluation Metrics")
    st.markdown("Below is the recorded training and validation performance across the 5 fine-tuning epochs:")
    
    # Exact values from your Google Colab training logs
    colab_metrics = pd.DataFrame({
        'Epoch': [1, 2, 3, 4, 5],
        'Train Loss': [0.2381, 0.1070, 0.0681, 0.0419, 0.0229],
        'Train Accuracy (%)': [91.63, 96.70, 97.78, 98.72, 99.43],
        'Val Loss': [0.1268, 0.0841, 0.0629, 0.0434, 0.0325],
        'Val Accuracy (%)': [96.21, 97.45, 97.71, 98.50, 99.21],
        'Val Precision (%)': [96.21, 97.45, 97.71, 98.51, 99.22],
        'Val Recall (%)': [96.22, 97.44, 97.71, 98.50, 99.20],
        'Val F1-Score (%)': [96.21, 97.45, 97.71, 98.50, 99.21]
    })
    st.dataframe(colab_metrics, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# PAGE 1: DIAGNOSTIC SCREENING
elif page == "🩻 Diagnostic Image Screening":
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

        # Model Prediction
        img_t = predict_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)[0]
            
        diseased_prob = float(probs[0])
        normal_prob = float(probs[1])
        
        stage_name, alert_class, clinical_guidelines = evaluate_severity(diseased_prob)

        # Auto-log to session history state
        if not ((st.session_state.history['Filename'] == uploaded_file.name).any()):
            new_entry = pd.DataFrame([{
                'Filename': uploaded_file.name,
                'Predicted': stage_name,
                'Ground Truth': stage_name,
                'Confidence': f"{max(diseased_prob, normal_prob)*100:.1f}%",
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)

        with col2:
            st.markdown('<div class="med-card">', unsafe_allow_html=True)
            st.subheader("Automated Analysis Output")
            st.progress(diseased_prob, text=f"Diabetic Retinopathy: {diseased_prob*100:.1f}%")
            st.progress(normal_prob, text=f"Healthy Retina: {normal_prob*100:.1f}%")
            
            # Color-Coded Staging Banner
            st.markdown(f'<div class="{alert_class}">Diagnostic Finding: {stage_name}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Management Guidelines
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.subheader("2. Clinical Staging Guidelines")
        st.text_area("Protocol Recommendations", value=clinical_guidelines, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

# PAGE 2: METRICS & CONFUSION MATRIX
elif page == "📊 Input Metrics & Confusion Matrix":
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("Final Model Validation Evaluation Metrics (Google Colab Final Epoch)")
    
    # Colab Final Validation Metrics Display
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "99.21%")
    m2.metric("Precision", "99.22%")
    m3.metric("Recall", "99.20%")
    m4.metric("F1-Score", "99.21%")
    st.markdown('</div>', unsafe_allow_html=True)

    # Confusion Matrix Section
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("Confusion Matrix (Class-Balanced Model)")
    
    # Array matching the exact Colab output values: TP=547, FN=7, FP=2, TN=580
    cm_colab = np.array([[547, 7],
                         [2, 580]])
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_colab, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Diseased', 'Normal'],
                yticklabels=['Diseased', 'Normal'], ax=ax)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Medical Label')
    plt.title('Google Colab Final Confusion Matrix')
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# PAGE 3: PATIENT RECORDS LOG
elif page == "📋 Patient Assessment Logs":
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("Historic Upload Logs")
    if len(st.session_state.history) > 0:
        st.dataframe(st.session_state.history, use_container_width=True)
    else:
        st.info("No saved records found.")
    st.markdown('</div>', unsafe_allow_html=True)
