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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetic Retinopathy Clinical Portal",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

TEAM_NAME = "ML--5th Floor--Group 3"
SUBMISSION_DATE = "Sept 8, 2026"
PROJECT_MODEL = "ResNet18 Transfer Learning"


# ============================================================
# CUSTOM CSS - WEBSITE DESIGN
# ============================================================

st.markdown("""
<style>

    /* =========================
       MAIN BACKGROUND
       ========================= */

    .stApp {
        background:
        linear-gradient(
            135deg,
            #F8FBFF 0%,
            #EEF6FF 50%,
            #F8FAFC 100%
        );
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* =========================
       HEADER
       ========================= */

    .header-box {
        background:
        linear-gradient(
            135deg,
            #0F2A5F 0%,
            #1E40AF 50%,
            #2563EB 100%
        );

        padding: 30px;
        border-radius: 20px;

        color: white;

        margin-bottom: 30px;

        box-shadow:
        0 12px 30px rgba(30,64,175,0.20);

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

        border-top:
        1px solid rgba(255,255,255,0.25);

        padding-top: 12px;
    }


    /* =========================
       SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {
        background:
        linear-gradient(
            180deg,
            #E0F2FE 0%,
            #EFF6FF 60%,
            #F8FAFC 100%
        );

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


    /* =========================
       HEADINGS
       ========================= */

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


    /* =========================
       INFORMATION CARDS
       ========================= */

    .info-card {
        background: rgba(255,255,255,0.92);

        padding: 24px;

        border-radius: 16px;

        border: 1px solid #DCE7F5;

        box-shadow:
        0 7px 22px rgba(15,42,95,0.08);

        margin-bottom: 20px;

        transition: all 0.25s ease;
    }

    .info-card:hover {
        transform: translateY(-3px);

        box-shadow:
        0 12px 30px rgba(15,42,95,0.12);
    }


    /* =========================
       MODEL CARD
       ========================= */

    .model-card {
        background:
        linear-gradient(
            145deg,
            #FFFFFF,
            #F0F7FF
        );

        padding: 25px;

        border-radius: 17px;

        border: 1px solid #BFDBFE;
[2026-09-08 11:20] ha'zel: box-shadow:
        0 8px 25px rgba(30,64,175,0.08);
    }

    .model-card-title {
        font-size: 1.25rem;

        font-weight: 800;

        color: #1E3A8A;

        margin-bottom: 15px;
    }

    .model-item {
        padding: 10px 0;

        border-bottom:
        1px solid #E5E7EB;

        font-size: 0.94rem;

        line-height: 1.5;
    }

    .model-item:last-child {
        border-bottom: none;
    }


    /* =========================
       METRIC CARDS
       ========================= */

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.95);

        border: 1px solid #D8E5F5;

        padding: 20px;

        border-radius: 15px;

        box-shadow:
        0 6px 18px rgba(15,42,95,0.07);

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


    /* =========================
       FILE UPLOADER
       ========================= */

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.95);

        border: 2px dashed #60A5FA;

        border-radius: 16px;

        padding: 15px;

        box-shadow:
        0 6px 20px rgba(37,99,235,0.07);

        transition: all 0.25s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #2563EB;

        background: #F8FBFF;

        box-shadow:
        0 10px 28px rgba(37,99,235,0.12);
    }


    /* =========================
       BUTTON
       ========================= */

    .stButton > button {
        width: 100%;

        background:
        linear-gradient(
            135deg,
            #1E40AF,
            #2563EB
        );

        color: white;

        border: none;

        border-radius: 11px;

        padding: 13px 22px;

        font-size: 1rem;

        font-weight: 700;

        box-shadow:
        0 6px 15px rgba(37,99,235,0.20);

        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        background:
        linear-gradient(
            135deg,
            #1D4ED8,
            #3B82F6
        );

        transform: translateY(-2px);

        box-shadow:
        0 10px 22px rgba(37,99,235,0.28);
    }


    /* =========================
       SELECT BOX
       ========================= */

    div[data-baseweb="select"] > div {
        border-radius: 10px !important;

        border:
        1px solid #CBD5E1 !important;

        background-color:
        white !important;
    }


    /* =========================
       DISEASE ALERT
       ========================= */

    .alert-stage3 {
        background:
        linear-gradient(
            135deg,
            #7F1D1D,
            #991B1B
        );

        color: #FFFFFF;

        border-left:
        6px solid #EF4444;

        padding: 20px;

        border-radius: 12px;

        font-weight: 600;

        margin-top: 15px;

        box-shadow:
        0 8px 22px rgba(127,29,29,0.20);
    }


    /* =========================
       IMAGE
       ========================= */

    [data-testid="stImage"] {
        background: white;

        padding: 10px;

        border-radius: 16px;

        border: 1px solid #DCE7F5;

        box-shadow:
        0 6px 20px rgba(15,42,95,0.08);
    }


    /* =========================
       CLINICAL INFORMATION
       ========================= */

    .clinical-note {
        background: #EFF6FF;

        border-left:
        5px solid #2563EB;

        padding: 18px;

        border-radius: 10px;

        color: #1E3A8A;

        margin-top: 20px;

        font-size: 0.92rem;

        line-height: 1.6;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer {
        margin-top: 50px;

        padding: 22px;

        text-align: center;

        color: #64748B;

        font-size: 0.82rem;
[2026-09-08 11:20] ha'zel: border-top:
        1px solid #DCE7F5;
    }

    .footer strong {
        color: #1E40AF;
    }


    /* =========================
       DIVIDERS
       ========================= */

    hr {
        border: none;

        border-top:
        1px solid #DCE7F5;

        margin: 25px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(f"""
<div class="header-box">

    <div class="main-title">
        👁️ Clinical Decision Support Portal
    </div>

    <div class="subtitle">
        Automated Diagnostic Assessment & Microvascular Evaluation
        | Ophthalmology AI Support
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


# ============================================================
# SESSION STATE
# ============================================================

if 'evaluation_data' not in st.session_state:

    # Exact test matrix:
    # 110 TP, 3 FN, 2 FP, 116 TN

    y_true_base = [1] * 113 + [0] * 118

    y_pred_base = (
        [1] * 110 +
        [0] * 3 +
        [1] * 2 +
        [0] * 116
    )

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


# ============================================================
# MODEL DOWNLOAD
# ============================================================

MODEL_FILE_ID = '1liKVBcah0zt-Yku3wIKJ20
