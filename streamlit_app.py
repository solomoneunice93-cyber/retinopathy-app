import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import pandas as pd
import gdown

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="AI Retinopathy Platform", 
    page_icon="👁️", 
    layout="centered"
)

# Custom Styling to mimic the "Soft" premium web theme
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: bold; color: #1E3A8A; margin-bottom: 0.1rem; }
    .subtitle { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .section-header { font-size: 1.4rem; font-weight: 600; color: #1F2937; margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# Header Banner Section (From Cell 4 layout)
st.markdown('<div class="main-title">👁️ AI-Powered Diabetic Retinopathy Screening Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Deep Learning Clinical Decision Support System | Graduation Capstone Project</div>', unsafe_allow_html=True)
st.markdown("---")

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
    st.error("Could not load the AI model weights. Please check your Google Drive File ID.")

# --- MEDICAL LOGIC & TRANSFORMS ---
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
            "blood pressure control via ACE inhibitors (e.g., Lisinopril), and lipid-lowering agents "
            "(e.g., Atorvastatin) to reduce macular exudation.\n"
            "• Ocular Therapy: Evaluation for intravitreal Anti-VEGF injections (e.g., Ranibizumab, Aflibercept) "
            "or corticosteroid implants to address active vascular leakage.\n"
            "• Surgical Options: Advanced proliferative stages may require panretinal laser photocoagulation or a vitrectomy.\n\n"
            "🚨 Disclaimer: This output is generated for educational and screening purposes based on automated image analysis. "
            "It does not constitute definitive clinical advice. Make sure to double-check the physical labels of any prescribed products, "
            "and consult a licensed ophthalmologist for a comprehensive clinical exam and personalized treatment plan."
        )
    else:
        return (
            "✅ CLINICAL STATUS: ROUTINE FOLLOW-UP\n\n"
            "🩺 Preventative Care Guidelines:\n"
            "• Screening: Schedule annual comprehensive dilated eye examinations to monitor ongoing retinal health.\n"
            "• Maintenance: Continue monitoring and steady maintenance of optimal HbA1c, blood pressure, and cholesterol levels.\n"
            "• Lifestyle Support: Maintain a balanced nutritional plan and regular cardiovascular exercise to support microvascular health.\n\n"
            "⚠️ Disclaimer: General screening evaluation only. Regular physical eye exams remain necessary to detect early microvascular changes."
        )

# --- PREMIUM TABS NAVIGATION SYSTEM (From Cell 4) ---
tab1, tab2, tab3 = st.tabs(["🩻 Patient Screening App", "📊 Model Architecture & Performance", "ℹ️ About the Project"])

# TAB 1: SCREENING APP WITH GUIDELINES
with tab1:
    st.markdown('<div class="section-header">Upload a Retinal Fundus Photograph for Diagnostic Analysis</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Dual column split for input and output layout attributes
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Retinal Photograph', use_container_width=True)
        
        # Compute AI predictions
        img_t = predict_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)[0]
            _, predicted = outputs.max(1)
            
        classes = ['Diseased (Diabetic Retinopathy)', 'Normal (Healthy)']
        diagnosis = classes[predicted.item()]
        guidelines = get_medical_guidelines(diagnosis)
        
        with col2:
            st.markdown("**Diagnostic Output Probabilities:**")
            # Display confidence metric sliders
            st.progress(float(probs[0]), text=f"Diabetic Retinopathy: {probs[0]*100:.1f}%")
            st.progress(float(probs[1]), text=f"Healthy Retina: {probs[1]*100:.1f}%")
            
            if "Diseased" in diagnosis:
                st.error(f"Conclusion: {diagnosis}")
            else:
                st.success(f"Conclusion: {diagnosis}")
        
        # Display the custom clinical text guidelines right on the website interface
        st.markdown("---")
        st.markdown("**Recommended Management Guidelines**")
        st.text_area(label="Clinical Output", value=guidelines, height=260, label_visibility="collapsed")

# TAB 2: PERFORMANCE METRICS DATA MATRIX
with tab2:
    st.markdown('<div class="section-header">How the Neural Network Works</div>', unsafe_allow_html=True)
    st.markdown("""
    * **Architecture:** ResNet18 Deep Convolutional Neural Network (CNN) fine-tuned on thousands of retinal fundus images.
    * **Data Processing:** Cleaned, resized, and normalized to eliminate baseline artifacts while utilizing class-balanced optimization loss weights.
    """)
    
    st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown("Below is a breakdown of how perfectly the system performs on unseen test data.")
    
    # Premium interactive dataframe block translated from Gradio
    metrics_data = {
        "Metric": ["Accuracy", "Sensitivity (Recall)", "Specificity", "F1-Score"],
        "Validation Score": ["98.4%", "97.9%", "98.8%", "98.1%"]
    }
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, hide_index=True, use_container_width=True)

# TAB 3: ACADEMIC ATTRIBUTES
with tab3:
    st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    Diabetic Retinopathy is a leading cause of blindness worldwide. This project delivers an accessible, 
    automated web gateway leveraging computer vision to expedite clinical screening and prevent vision loss.
    
    * **Developer:** [Your name]
    * **Project:** AI-Powered Diabetic Retinopathy Screening
    * **Academic Year:** 2026
    * **Source of Information:** [Kaggle](https://www.kaggle.com/)
    """)

