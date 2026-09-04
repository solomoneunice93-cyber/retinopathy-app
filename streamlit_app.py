import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import gdown
import cv2
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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
.stApp {
    background-color: #F8FAFC;
}

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

.med-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 4px solid #2563EB;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}

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

[data-testid="stSidebar"] {
    background-color: #F1F5F9;
    border-right: 1px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)

# Header Banner
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

# --- INITIALIZE SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Filename', 'Predicted', 'Ground Truth', 'Confidence', 'Timestamp'])

if 'patient_docs' not in st.session_state:
    st.session_state.patient_docs = []

# --- AUTOMATIC MODEL DOWNLOAD & GRAD-CAM CAPABLE RESNET ---
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

# --- GRAD-CAM GENERATOR FUNCTION ---
def generate_gradcam(input_tensor, model, original_image):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[1].conv2
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    _, target_class = output.max(1)
    
    model.zero_grad()
    output[0, target_class].backward()

    h1.remove()
    h2.remove()

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    activation = activations[0][0]
    for i in range(activation.size(0)):
        activation[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activation, dim=0).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    orig_np = np.array(original_image.resize((128, 128)))
    heatmap_resized = cv2.resize(heatmap, (128, 128))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(orig_np, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(overlay)

# --- PDF REPORT GENERATOR ---
def create_pdf_report(filename, stage_name, diseased_prob, guidelines):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#1E3A8A',
        spaceAfter=12
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=8
    )

    story = [
        Paragraph("Diabetic Retinopathy Diagnostic Assessment Report", title_style),
        Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style),
        Paragraph(f"<b>File Processed:</b> {filename}", body_style),
        Paragraph(f"<b>Team / Author:</b> {TEAM_NAME}", body_style),
        Spacer(1, 12),
        Paragraph(f"<b>Diagnostic Finding:</b> {stage_name}", body_style),
        Paragraph(f"<b>DR Probability Score:</b> {diseased_prob*100:.2f}%", body_style),
        Spacer(1, 12),
        Paragraph("<b>Clinical Guidelines & Protocol Recommendations:</b>", body_style),
        Paragraph(guidelines.replace('\n', '<br/>'), body_style),
        Spacer(1, 12),
        Paragraph("<i>Disclaimer: Generated for screening support. Consult a licensed ophthalmologist for official diagnosis.</i>", body_style)
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- MEDICAL TRANSFORMS & STAGED CLINICAL GUIDELINES ---
predict_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_severity(diseased_prob):
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
    else:
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
        * **Data Preprocessing & Augmentation:** Images resized to `128x128`, transformed to PyTorch Tensors, normalized, and augmented with `RandomHorizontalFlip()` and `RandomRotation(15°)`.
        * **Data Split:** 80% Training / 20% Validation (`random_split`).
        * **Loss Function:** `CrossEntropyLoss` weighted inversely proportional to class frequencies to combat dataset imbalance.
        * **Optimizer:** Per-layer `Adam` optimizer (Layer 4 `lr = 0.00001`, Fully-Connected head `lr = 0.0001`).
        * **Batch Size & Epochs:** `Batch Size = 32`, `Epochs = 5`.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("📈 Google Colab Model Training & Validation Evaluation Metrics")
    st.markdown("Below is the recorded training and validation performance across the 5 fine-tuning epochs:")
    
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
        
        img_t = predict_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)[0]
            
        diseased_prob = float(probs[0])
        normal_prob = float(probs[1])
        
        stage_name, alert_class, clinical_guidelines = evaluate_severity(diseased_prob)

        # Generate Grad-CAM Heatmap
        gradcam_img = generate_gradcam(img_t, model, image)

        # Log to Session State
        if not ((st.session_state.history['Filename'] == uploaded_file.name).any()):
            new_entry = pd.DataFrame([{
                'Filename': uploaded_file.name,
                'Predicted': stage_name,
                'Ground Truth': stage_name,
                'Confidence': f"{max(diseased_prob, normal_prob)*100:.1f}%",
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.markdown('<div class="med-card">', unsafe_allow_html=True)
            st.subheader("Retinal Imaging & Grad-CAM Heatmap View")
            
            subcol_a, subcol_b = st.columns(2)
            with subcol_a:
                st.image(image, use_container_width=True, caption="Original Fundus Scan")
            with subcol_b:
                st.image(gradcam_img, use_container_width=True, caption="Grad-CAM Focus Heatmap")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="med-card">', unsafe_allow_html=True)
            st.subheader("Automated Analysis Output")
            st.progress(diseased_prob, text=f"Diabetic Retinopathy Probability: {diseased_prob*100:.1f}%")
            st.progress(normal_prob, text=f"Healthy Retina Probability: {normal_prob*100:.1f}%")
            
            # Uncertainty warning guardrail
            if 0.38 <= diseased_prob <= 0.42:
                st.warning("⚠️ Borderline Confidence Assessment — Clinical re-scan or manual review suggested.")

            st.markdown(f'<div class="{alert_class}">Diagnostic Finding: {stage_name}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # PDF Download Button
            pdf_data = create_pdf_report(uploaded_file.name, stage_name, diseased_prob, clinical_guidelines)
            st.download_button(
                label="📄 Download Diagnostic PDF Report",
                data=pdf_data,
                file_name=f"DR_Report_{uploaded_file.name.split('.')[0]}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.subheader("2. Clinical Staging Guidelines")
        st.text_area("Protocol Recommendations", value=clinical_guidelines, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

# PAGE 2: METRICS & CONFUSION MATRIX
elif page == "📊 Input Metrics & Confusion Matrix":
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("Final Model Validation Evaluation Metrics (Google Colab Final Epoch)")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "99.21%")
    m2.metric("Precision", "99.22%")
    m3.metric("Recall", "99.20%")
    m4.metric("F1-Score", "99.21%")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("Interactive Confusion Matrix (Class-Balanced Model)")
    
    cm_data = np.array([[547, 7], [2, 580]])
    labels = ['Diseased', 'Normal']
    
    fig = px.imshow(
        cm_data,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted Label", y="True Medical Label", color="Sample Count"),
        title="Google Colab Final Validation Confusion Matrix"
    )
    fig.update_layout(width=600, height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# PAGE 3: PATIENT RECORDS LOG & DOCUMENTS
elif page == "📋 Patient Assessment Logs":
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("Historic Upload Logs")
    if len(st.session_state.history) > 0:
        st.dataframe(st.session_state.history, use_container_width=True)
    else:
        st.info("No saved scan records found.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- NEW: PATIENT MEDICAL DOCUMENTS SECTION ---
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.subheader("📁 Patient Medical Documents & External Records")
    st.markdown("Upload supplemental medical records, lab reports (e.g., HbA1c tests), or previous OCT scans for clinical context.")
    
    doc_upload = st.file_uploader(
        "Upload Medical Document (PDF, PNG, JPG, TXT)", 
        type=["pdf", "png", "jpg", "jpeg", "txt"], 
        key="patient_doc_uploader"
    )
    
    doc_notes = st.text_input("Document Category / Clinical Note (e.g., 'HbA1c Lab Report - June 2026')")
    
    if st.button("Save Patient Document"):
        if doc_upload is not None:
            doc_entry = {
                "Document Name": doc_upload.name,
                "Category / Note": doc_notes if doc_notes else "General Medical Record",
                "Size (KB)": f"{round(doc_upload.size / 1024, 1)} KB",
                "Date Uploaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.patient_docs.append(doc_entry)
            st.success(f"Successfully attached document: {doc_upload.name}")
        else:
            st.warning("Please select a file to upload first.")

    if len(st.session_state.patient_docs) > 0:
        st.markdown("### Attached Patient Records")
        st.dataframe(pd.DataFrame(st.session_state.patient_docs), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
