import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import gdown

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Retinopathy Assistant", page_icon="👁️", layout="centered")
st.title("👁️ AI Retinopathy Assistant with Clinical Guidance")
st.write("Upload a retinal photograph to see the predictive diagnosis alongside standard clinical management options.")

# --- AUTOMATIC MODEL DOWNLOAD ---
# Replace 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE' with your actual Google Drive ID
MODEL_FILE_ID = '1liKVBcah0zt-Yku3wIKJ20_idwwcEmh0/view?usp=sharing' 
MODEL_PATH = "diabetic_retinopathy_resnet18.pth"

@st.cache_resource
def load_medical_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://google.com{MODEL_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    
    # Recreate architecture
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

# --- TRANSFORMS & UTILS ---
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

# --- USER INTERFACE ---
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Retinal Photograph', use_container_width=True)
    
    # Process and Predict
    img_t = predict_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = outputs.max(1)
        
    classes = ['Diseased (Diabetic Retinopathy)', 'Normal (Healthy)']
    diagnosis = classes[predicted.item()]
    guidelines = get_medical_guidelines(diagnosis)
    
    # Display Results
    st.subheader("Diagnostic Conclusion")
    if "Diseased" in diagnosis:
        st.error(diagnosis)
    else:
        st.success(diagnosis)
        
    st.subheader("Recommended Management Guidelines")
    st.text_area(label="Clinical Output", value=guidelines, height=300, label_visibility="collapsed")
