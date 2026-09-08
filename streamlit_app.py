[9/8/2026 5:54 AM] ha'zel: import os
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

st.set_page_config(
    page_title="Diabetic Retinopathy Clinical Portal",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

TEAM_NAME = "ML--5th Floor--Group 3"
SUBMISSION_DATE = "Sept 8, 2026"
PROJECT_MODEL = "ResNet18 Transfer Learning"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F8FBFF 0%, #EEF6FF 50%, #F8FAFC 100%);
    color: #1E293B;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.header-box {
    background: linear-gradient(135deg, #0F2A5F 0%, #1E40AF 50%, #2563EB 100%);
    padding: 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 12px 30px rgba(30,64,175,0.20);
    border: 1px solid rgba(255,255,255,0.15);
}

.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 8px;
    letter-spacing: -0.7px;
}

.subtitle {
    font-size: 1.05rem;
    opacity: 0.92;
    margin-bottom: 15px;
    line-height: 1.6;
}

.team-meta {
    font-size: 0.85rem;
    color: #DBEAFE;
    border-top: 1px solid rgba(255,255,255,0.25);
    padding-top: 12px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E0F2FE 0%, #EFF6FF 60%, #F8FAFC 100%);
    border-right: 1px solid #BFDBFE;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #0F2A5F;
    font-weight: 800;
}

[data-testid="stSidebar"] label {
    color: #334155;
    font-weight: 600;
}

h1 {
    color: #0F2A5F !important;
    font-weight: 800 !important;
}

h2 {
    color: #163B72 !important;
    font-weight: 750 !important;
}

h3 {
    color: #1E40AF !important;
    font-weight: 700 !important;
}

.info-card {
    background: rgba(255,255,255,0.92);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #DCE7F5;
    box-shadow: 0 7px 22px rgba(15,42,95,0.08);
    margin-bottom: 20px;
    transition: all 0.25s ease;
}

.info-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(15,42,95,0.12);
}

.model-card {
    background: linear-gradient(145deg, #FFFFFF, #F0F7FF);
    padding: 25px;
    border-radius: 17px;
    border: 1px solid #BFDBFE;
    box-shadow: 0 8px 25px rgba(30,64,175,0.08);
}

.model-card-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1E3A8A;
    margin-bottom: 15px;
}

.model-item {
    padding: 10px 0;
    border-bottom: 1px solid #E5E7EB;
    font-size: 0.94rem;
    line-height: 1.5;
}

.model-item:last-child {
    border-bottom: none;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.95);
    border: 1px solid #D8E5F5;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 6px 18px rgba(15,42,95,0.07);
    transition: transform 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #1E40AF !important;
    font-weight: 800;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.95);
    border: 2px dashed #60A5FA;
    border-radius: 16px;
    padding: 15px;
    box-shadow: 0 6px 20px rgba(37,99,235,0.07);
    transition: all 0.25s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #2563EB;
    background: #F8FBFF;
    box-shadow: 0 10px 28px rgba(37,99,235,0.12);
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1E40AF, #2563EB);
    color: white;
    border: none;
    border-radius: 11px;
    padding: 13px 22px;
    font-size: 1rem;
    font-weight: 700;
    box-shadow: 0 6px 15px rgba(37,99,235,0.20);
    transition: all 0.25s ease;
}
[9/8/2026 5:54 AM] ha'zel: .stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(37,99,235,0.28);
}

div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1px solid #CBD5E1 !important;
    background-color: white !important;
}

.alert-stage3 {
    background: linear-gradient(135deg, #7F1D1D, #991B1B);
    color: #FFFFFF;
    border-left: 6px solid #EF4444;
    padding: 20px;
    border-radius: 12px;
    font-weight: 600;
    margin-top: 15px;
    box-shadow: 0 8px 22px rgba(127,29,29,0.20);
}

[data-testid="stAlert"] {
    border-radius: 12px !important;
}

[data-testid="stImage"] {
    background: white;
    padding: 10px;
    border-radius: 16px;
    border: 1px solid #DCE7F5;
    box-shadow: 0 6px 20px rgba(15,42,95,0.08);
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #DCE7F5;
    box-shadow: 0 6px 20px rgba(15,42,95,0.07);
}

.clinical-note {
    background: #EFF6FF;
    border-left: 5px solid #2563EB;
    padding: 18px;
    border-radius: 10px;
    color: #1E3A8A;
    margin-top: 20px;
    font-size: 0.92rem;
    line-height: 1.6;
}

.footer {
    margin-top: 50px;
    padding: 22px;
    text-align: center;
    color: #64748B;
    font-size: 0.82rem;
    border-top: 1px solid #DCE7F5;
}

.footer strong {
    color: #1E40AF;
}

hr {
    border: none;
    border-top: 1px solid #DCE7F5;
    margin: 25px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header-box">
    <div class="main-title">👁️ Clinical Decision Support Portal</div>
    <div class="subtitle">
        Automated Diagnostic Assessment & Microvascular Evaluation |
        Ophthalmology AI Support
    </div>
    <div class="team-meta">
        <strong>Developed by:</strong> {TEAM_NAME}
        &nbsp; | &nbsp;
        <strong>Submission Date:</strong> {SUBMISSION_DATE}
        &nbsp; | &nbsp;
        <strong>Model Architecture:</strong> {PROJECT_MODEL}
    </div>
</div>
""", unsafe_allow_html=True)

if 'evaluation_data' not in st.session_state:
    y_true_base = [1] * 113 + [0] * 118
    y_pred_base = [1] * 110 + [0] * 3 + [1] * 2 + [0] * 116
    st.session_state.evaluation_data = {
        'y_true': y_true_base,
        'y_pred': y_pred_base
    }

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=[
            'Filename',
            'Predicted',
            'Ground Truth',
            'Confidence',
            'Timestamp'
        ]
    )

if 'patient_docs' not in st.session_state:
    st.session_state.patient_docs = []

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

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device('cpu')
        )
    )

    model.eval()
    return model

try:
    model = load_medical_model()
except Exception as e:
    st.error(
        "⚠️ Model Loading Error: Unable to fetch model weights from Google Drive."
    )
    model = None

def generate_gradcam(input_tensor, model, original_image):
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        if output.requires_grad:
            output.register_hook(save_gradient)

    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    target_layer = model.layer4[1].conv2
    hook = target_layer.register_forward_hook(forward_hook)
[9/8/2026 5:54 AM] ha'zel: with torch.set_grad_enabled(True):
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        model.zero_grad()
        loss = output[0, pred_class]
        loss.backward()

    hook.remove()

    if not gradients or not activations:
        st.error("Failed to extract Grad-CAM features.")
        return np.array(original_image), pred_class, 0.0

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(
        acts.shape[1:],
        dtype=np.float32
    )

    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]

    cam = np.maximum(cam, 0)

    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    cam = cv2.resize(
        cam,
        (
            original_image.width,
            original_image.height
        )
    )

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    orig_np = np.array(original_image)

    overlay = cv2.addWeighted(
        orig_np,
        0.6,
        heatmap,
        0.4,
        0
    )

    probabilities = torch.softmax(output, dim=1)
    confidence = probabilities[0, pred_class].item()

    return (
        overlay,
        pred_class,
        confidence
    )

st.sidebar.header("Clinical Navigation")

view_selection = st.sidebar.radio(
    "Select View:",
    [
        "Overview & Model Architecture",
        "Diagnostic Image Screening",
        "Input Metrics & Confusion Matrix",
        "Patient Assessment Logs"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"Team: {TEAM_NAME}")
st.sidebar.markdown(f"Date: {SUBMISSION_DATE}")

if view_selection == "Overview & Model Architecture":

    st.header("1. System Overview & Model Architecture")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown("""
        <div class="info-card">
            <h3>👁️ Diabetic Retinopathy AI Diagnostic System</h3>
            <p>
            This clinical decision support portal leverages a
            fine-tuned ResNet-18 Deep Convolutional Neural Network
            trained to detect microvascular lesions, cotton wool spots,
            hemorrhages, and exudates in retinal fundus photographs.
            </p>
            <p>
            The model processes high-resolution ocular scans,
            classifies images into Healthy or Diseased
            (Diabetic Retinopathy) states, and computes visual
            explainability maps using Grad-CAM to highlight relevant
            pathology.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Key Technical Features")

        st.markdown("""
        <div class="info-card">
            <strong>🧠 Deep Feature Extraction</strong><br>
            ResNet-18 residual connections prevent vanishing gradients
            during deep feature analysis.
        </div>

        <div class="info-card">
            <strong>🔥 Visual Explainability</strong><br>
            Integrated Grad-CAM maps highlight retinal regions
            influencing model classification.
        </div>

        <div class="info-card">
            <strong>📊 Real-time Metrics Update</strong><br>
            Dynamic updates to the underlying confusion matrix
            upon new diagnostic verification.
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="model-card">
            <div class="model-card-title">
                🧠 Model Specifications
            </div>

            <div class="model-item">
                <strong>Architecture:</strong><br>
                ResNet-18
            </div>

            <div class="model-item">
                <strong>Input Size:</strong><br>
                224 × 224 × 3
            </div>
[9/8/2026 5:54 AM] ha'zel: <div class="model-item">
                <strong>Classifier:</strong><br>
                FC(512 → 256) → ReLU → Dropout(0.4) → FC(256 → 2)
            </div>

            <div class="model-item">
                <strong>Framework:</strong><br>
                PyTorch & Streamlit
            </div>

            <div class="model-item">
                <strong>Explainability:</strong><br>
                Grad-CAM
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="clinical-note">
        <strong>🔬 Clinical AI Support</strong><br><br>
        This system analyzes retinal fundus images using a
        deep-learning model and provides a visual Grad-CAM explanation
        of the regions influencing the prediction.
        <br><br>
        <strong>Important:</strong>
        This system is intended for research and clinical
        decision-support purposes and should not replace professional
        ophthalmological examination.
    </div>
    """, unsafe_allow_html=True)

elif view_selection == "Diagnostic Image Screening":

    st.header("1. Retinal Fundus Image Upload")

    st.write("Upload Retinal Scan (JPG, PNG)")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    ground_truth_selection = "Diseased"

    if uploaded_file is not None:

        col_img1, col_img2 = st.columns(2)

        original_img = Image.open(
            uploaded_file
        ).convert("RGB")

        with col_img1:

            st.image(
                original_img,
                caption="Uploaded Retinal Fundus Scan",
                use_container_width=True
            )

        ground_truth_selection = st.selectbox(
            "Ground Truth Label (Verification):",
            ["Diseased", "Healthy"]
        )

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        input_tensor = transform(
            original_img
        ).unsqueeze(0)

        if st.button("Run Diagnostic Assessment"):

            if model is None:

                st.error(
                    "Model is not available."
                )

            else:

                with st.spinner(
                    "Processing image and generating Grad-CAM activation map..."
                ):

                    overlay, pred_class, confidence = generate_gradcam(
                        input_tensor,
                        model,
                        original_img
                    )

                with col_img2:

                    st.image(
                        overlay,
                        caption="Grad-CAM Pathology Map",
                        use_container_width=True
                    )

                pred_label = (
                    "Diseased"
                    if pred_class == 1
                    else "Healthy"
                )

                st.markdown("---")

                st.subheader(
                    "Diagnostic Results & Clinical Recommendations"
                )

                res_col1, res_col2 = st.columns(2)

                with res_col1:

                    st.metric(
                        "Predicted Condition",
                        pred_label
                    )

                with res_col2:

                    st.metric(
                        "Model Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                st.markdown(
                    "### Clinical Action & Recommendations"
                )

                if pred_class == 1:

                    st.markdown("""
                    <div class="alert-stage3">

                        ⚠️ <strong>
                        ALERT: Pathological Microvascular Lesions Detected
                        </strong>
[9/8/2026 5:54 AM] ha'zel: <br><br>

                        <strong>Recommended Next Steps:</strong>

                        <ul>
                            <li>
                            Schedule an urgent referral to an
                            Ophthalmology Specialist / Retina Clinic.
                            </li>

                            <li>
                            Perform Optical Coherence Tomography (OCT)
                            to evaluate potential Macular Edema.
                            </li>

                            <li>
                            Advise patient on tight glycemic and
                            blood pressure control.
                            </li>
                        </ul>

                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.success("""
                    ✅ No Pathological DR Features Detected

                    * Recommendation: Routine annual retinal screening recommended.
                    * Continue standard diabetes care and blood glucose monitoring.
                    """)

                true_class_int = (
                    1
                    if ground_truth_selection == "Diseased"
                    else 0
                )

                st.session_state.evaluation_data[
                    'y_true'
                ].append(true_class_int)

                st.session_state.evaluation_data[
                    'y_pred'
                ].append(pred_class)

                new_entry = pd.DataFrame([{
                    'Filename': uploaded_file.name,
                    'Predicted': pred_label,
                    'Ground Truth': ground_truth_selection,
                    'Confidence': f"{confidence * 100:.2f}%",
                    'Timestamp': pd.Timestamp.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                }])

                st.session_state.history = pd.concat(
                    [
                        st.session_state.history,
                        new_entry
                    ],
                    ignore_index=True
                )

elif view_selection == "Input Metrics & Confusion Matrix":

    st.header(
        "3. Clinical Metrics & Validation Performance"
    )

    y_true = np.array(
        st.session_state.evaluation_data['y_true']
    )

    y_pred = np.array(
        st.session_state.evaluation_data['y_pred']
    )

    tp = np.sum(
        (y_true == 1) &
        (y_pred == 1)
    )

    fp = np.sum(
        (y_true == 0) &
        (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1) &
        (y_pred == 0)
    )

    tn = np.sum(
        (y_true == 0) &
        (y_pred == 0)
    )

    accuracy = (
        (tp + tn) / len(y_true)
        if len(y_true) > 0
        else 0
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Overall Accuracy",
        f"{accuracy * 100:.1f}%"
    )

    m2.metric(
        "Sensitivity (Recall)",
        f"{sensitivity * 100:.1f}%"
    )

    m3.metric(
        "Specificity",
        f"{specificity * 100:.1f}%"
    )

    st.subheader("Confusion Matrix")

    cm = [
        [tn, fp],
        [fn, tp]
    ]

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(
            x="Predicted Label",
            y="True Label",
            color="Count"
        ),
        x=[
            'Healthy (0)',
            'Diseased (1)'
        ],
        y=[
            'Healthy (0)',
            'Diseased (1)'
        ],
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif view_selection == "Patient Assessment Logs":

    st.header(
        "4. Assessment History & Patient Logs"
    )

    if len(st.session_state.history) > 0:
[9/8/2026 5:54 AM] ha'zel: st.dataframe(
            st.session_state.history,
            use_container_width=True
        )

    else:

        st.info(
            "No patient scans have been processed in this session yet."
        )

st.markdown("""
<div class="footer">

    <strong>
    👁️ Diabetic Retinopathy Clinical Decision Support Portal
    </strong>

    <br><br>

    ResNet-18 Transfer Learning
    • Grad-CAM Explainability
    • PyTorch
    • Streamlit

    <br><br>

    Developed by
    <strong>ML--5th Floor--Group 3</strong>
    • September 2026

</div>
""", unsafe_allow_html=True)
