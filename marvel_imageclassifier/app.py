import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# MARVEL STREAMLIT APP
# ===============================

st.set_page_config(
    page_title="Marvel Character Classifier",
    page_icon="🦸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Marvel-style CSS
st.markdown("""
    <style>
        body {
            background-color: #1a1a1a;
            color: white;
        }
        .title {
            font-size: 50px;
            font-weight: bold;
            color: #E23636;
            text-align: center;
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 10px #E23636; }
            to { text-shadow: 0 0 30px #E23636, 0 0 10px #fff; }
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            margin-bottom: 30px;
            color: #f5f5f5;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL + CLASSES
# ===============================
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)  # 4 classes

    checkpoint = torch.load("models/marvel_resnet18.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["classes"]

model, class_names = load_model()

# ===============================
# IMAGE TRANSFORMS
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
    return probs

# ===============================
# UI LAYOUT
# ===============================
st.markdown("<div class='title'>MARVEL HERO CLASSIFIER</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image to reveal your hero 🦸‍♂️</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    probs = predict(image)
    pred_idx = np.argmax(probs)
    pred_class = class_names[pred_idx]

    st.subheader(f"✨ Predicted Hero: **{pred_class}**")
    st.write(f"Confidence: **{probs[pred_idx]*100:.2f}%**")

    # Confidence Bar Chart
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(class_names, probs, color=["#E23636" if i==pred_idx else "#888888" for i in range(len(class_names))])
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence per Class")
    st.pyplot(fig)
