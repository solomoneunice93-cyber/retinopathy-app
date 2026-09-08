import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import gdown
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Diabetic Retinopathy Clinical Portal",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS ---
TEAM_NAME = "ML--5th Floor--Group 3"
SUBMISSION_DATE = "Sept 8, 2026"
PROJECT_MODEL = "ResNet18 Transfer Learning"

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 12px;
    }
    .team-meta {
        font-size: 0.85rem;
        color: #E0F2FE;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        padding-top: 8px;
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
        background-color: #E0F2FE;
        border-right: 1px solid #BFDBFE;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER BANNER ---
st.markdown(f"""
<div class="header-box">
    <div class="main-title">👁️ Clinical Decision Support Portal</div>
    <div class="subtitle">Automated Diagnostic Assessment & Microvascular Evaluation | Ophthalmology AI Support</div>
    <div class="team-meta">
        <strong>Developed by:</strong> {TEAM_NAME} | <strong>Submission Date:</strong> {SUBMISSION_DATE} | <strong>Model Architecture:</strong> {PROJECT_MODEL}
    </div>
</div>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE FOR HISTORIC & EVALUATION DATA ---
if 'evaluation_data' not in st.session_state:
    # Matches exact Colab Test Matrix: 110 TP, 3 FN, 2 FP, 116 TN
    y_true_base = [1]*113 + [0]*118
    y_pred_base = [1]*110 + [0]*3 + [1]*2 + [0]*116
    st.session_state.evaluation_data = {'y_true': y_true_base, 'y_pred': y_pred_base}

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

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    # Hook into the final convolutional layer of ResNet18
    target_layer = model.layer4[1].conv2
    hook = target_layer.register_forward_hook(forward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward()

    hook.remove()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]

    cam = np.maximum(cam, 0)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    
    cam = cv2.resize(cam, (original_image.width, original_image.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    orig_np = np.array(original_image)
    overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
    
    return overlay, pred_class, torch.softmax(output, dim=1)[0, pred_class].item()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Clinical Navigation")
view_selection = st.sidebar.radio(
    "Select View:",
    ["Overview & Model Architecture", "Diagnostic Image Screening", "Input Metrics & Confusion Matrix", "Patient Assessment Logs"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Team:** {TEAM_NAME}")
st.sidebar.markdown(f"**Date:** {SUBMISSION_DATE}")

# --- VIEW 1: OVERVIEW & ARCHITECTURE ---
if view_selection == "Overview & Model Architecture":
    st.header("1. System Overview & Model Architecture")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Diabetic Retinopathy AI Diagnostic System")
        st.write("""
        This clinical decision support portal leverages a fine-tuned **ResNet-18 Deep Convolutional Neural Network** 
        trained to detect microvascular lesions, cotton wool spots, hemorrhages, and exudates in retinal fundus photographs.
        
        The model processes high-resolution ocular scans, classifies images into **Healthy** or **Diseased (Diabetic Retinopathy)** states, 
        and computes visual explainability maps using **Grad-CAM (Gradient-weighted Class Activation Mapping)** to highlight relevant pathology.
        """)
        
        st.markdown("### Key Technical Features")
        st.markdown("* **Deep Feature Extraction:** ResNet-18 residual connections prevent vanishing gradients during deep feature analysis.")
        st.markdown("* **Visual Explainability:** Integrated Grad-CAM maps highlight exact retinal regions influencing model classification.")
        st.markdown("* **Real-time Metrics Update:** Dynamic updates to the underlying confusion matrix upon new diagnostic verification.")

    with col2:
        st.info("### Model Specifications")
        st.markdown("**Architecture:** ResNet-18")
        st.markdown("**Input Size:** 224 x 224 x 3")
        st.markdown("**Classifier:** FC(512 → 256) → ReLU → Dropout(0.4) → FC(256 → 2)")
        st.markdown("**Framework:** PyTorch & Streamlit")

# --- VIEW 2: DIAGNOSTIC IMAGE SCREENING ---
elif view_selection == "Diagnostic Image Screening":
    st.header("1. Retinal Fundus Image Upload")
    st.write("Upload Retinal Scan (JPG, PNG)")
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # Fixed: Define default state before conditional checks to avoid NameError
    ground_truth_selection = "Diseased"

    if uploaded_file is not None:
        col_img1, col_img2 = st.columns(2)
        
        original_img = Image.open(uploaded_file).convert("RGB")
        with col_img1:
            st.image(original_img, caption="Uploaded Retinal Fundus Scan", use_container_width=True)

        ground_truth_selection = st.selectbox("Ground Truth Label (Verification):", ["Diseased", "Healthy"])

        # Preprocess Image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(original_img).unsqueeze(0)

        if st.button("Run Diagnostic Assessment"):
            with st.spinner("Processing image and generating Grad-CAM activation map..."):
                overlay, pred_class, confidence = generate_gradcam(input_tensor, model, original_img)

            with col_img2:
                st.image(overlay, caption="Grad-CAM Pathology Map", use_container_width=True)

            pred_label = "Diseased" if pred_class == 1 else "Healthy"
            
            st.markdown("---")
            st.subheader("Diagnostic Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Predicted Condition", pred_label)
            with res_col2:
                st.metric("Model Confidence", f"{confidence * 100:.2f}%")

            if pred_class == 1:
                st.markdown("""
                <div class="alert-stage3">
                    ⚠️ ALERT: Pathological lesions detected. Referral to an Ophthalmology Specialist recommended.
                </div>
                """, unsafe_allow_html=True)

            # Update Session State Evaluation Data
            true_class_int = 1 if 'ground_truth_selection' in locals() and ground_truth_selection == "Diseased" else 0
            st.session_state.evaluation_data['y_true'].append(true_class_int)
            st.session_state.evaluation_data['y_pred'].append(pred_class)

            # Update Session History
            new_entry = pd.DataFrame([{
                'Filename': uploaded_file.name,
                'Predicted': pred_label,
                'Ground Truth': ground_truth_selection,
                'Confidence': f"{confidence * 100:.2f}%",
                'Timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)

# --- VIEW 3: METRICS & CONFUSION MATRIX ---
elif view_selection == "Input Metrics & Confusion Matrix":
    st.header("3. Clinical Metrics & Validation Performance")

    y_true = np.array(st.session_state.evaluation_data['y_true'])
    y_pred = np.array(st.session_state.evaluation_data['y_pred'])

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Accuracy", f"{accuracy * 100:.1f}%")
    m2.metric("Sensitivity (Recall)", f"{sensitivity * 100:.1f}%")
    m3.metric("Specificity", f"{specificity * 100:.1f}%")

    st.subheader("Confusion Matrix")
    
    cm = [[tn, fp], [fn, tp]]
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=['Healthy (0)', 'Diseased (1)'],
        y=['Healthy (0)', 'Diseased (1)'],
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- VIEW 4: PATIENT LOGS ---
elif view_selection == "Patient Assessment Logs":
    st.header("4. Assessment History & Patient Logs")
    
    if len(st.session_state.history) > 0:
        st.dataframe(st.session_state.history, use_container_width=True)
    else:
        st.info("No patient scans have been processed in this session yet.")
