import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import gdown

# ---------------- Helper Function ---------------- #
def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# ---------------- Streamlit Page Setup ---------------- #
st.set_page_config(page_title="AI Builders 2025: Fake Social Media Detector", layout="centered")
st.title("🤖 AI Builders 2025: Fake social media account/profile detection")

# ---------------- IG (Tabular) Model ---------------- #
with st.expander("Fake/Bot IG account detector"):
    model = joblib.load("model/fake_ig_model.pkl")
    image = Image.open("sample_images/Screenshot 2568-06-04 at 17.45.14.png")
    st.image(image)
    st.markdown("#### Feature Descriptions")
    st.markdown("- **Bio Length**: Number of characters in the bio")
    st.markdown("<h4><b>Enter Instagram profile information below:</b></h4>", unsafe_allow_html=True)

    username = st.text_input("Username", value="user123")
    fullname = st.text_input("Full Name", value="John Doe")
    description_length = st.number_input("Bio Description Length", min_value=0)
    has_profile_pic = st.selectbox("Has Profile Picture?", ["Yes", "No"])
    external_url = st.selectbox("Has External URL?", ["Yes", "No"])
    is_private = st.selectbox("Is Private Account?", ["Yes", "No"])
    post_count = st.number_input("Number of Posts", min_value=0)
    follower_count = st.number_input("Number of Followers", min_value=0)
    following_count = st.number_input("Number of Following", min_value=0)

    username_digit_ratio = sum(c.isdigit() for c in username) / len(username) if len(username) > 0 else 0
    fullname_words = len(fullname.strip().split())
    fullname_digit_ratio = sum(c.isdigit() for c in fullname) / len(fullname.replace(" ", "")) if len(fullname.replace(" ", "")) > 0 else 0
    name_equals_username = 1 if fullname.replace(" ", "").lower() == username.lower() else 0

    log1p_followers = np.log1p(follower_count)
    log1p_following = np.log1p(following_count)
    log1p_posts = np.log1p(post_count)
    log1p_username_ratio = np.log1p(username_digit_ratio)
    log1p_fullname_ratio = np.log1p(fullname_digit_ratio)
    log1p_desc_len = np.log1p(description_length)
    follow_ratio = following_count / follower_count if follower_count > 0 else 0

    has_profile_pic = 1 if has_profile_pic == "Yes" else 0
    external_url = 1 if external_url == "Yes" else 0
    is_private = 1 if is_private == "Yes" else 0

    input_data = pd.DataFrame([[
        has_profile_pic,
        username_digit_ratio,
        fullname_words,
        fullname_digit_ratio,
        name_equals_username,
        description_length,
        external_url,
        is_private,
        post_count,
        follower_count,
        following_count,
        log1p_followers,
        log1p_following,
        log1p_posts,
        log1p_username_ratio,
        log1p_fullname_ratio,
        log1p_desc_len,
        follow_ratio
    ]], columns=[
        'profile pic',
        'nums/length username',
        'fullname words',
        'nums/length fullname',
        'name==username',
        'description length',
        'external URL',
        'private',
        '#posts',
        '#followers',
        '#follows',
        'log1p_#followers',
        'log1p_#follows',
        'log1p_#posts',
        'log1p_nums/length username',
        'log1p_nums/length fullname',
        'log1p_description length',
        'Follow_to_Followers_Ratio'
    ])

    if st.button("Detect Fake Account"):
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][prediction]
        if prediction == 1:
            st.error(f"Prediction: FAKE/BOT Account (Confidence: {proba:.2%})")
        else:
            st.success(f"Prediction: REAL Account (Confidence: {proba:.2%})")

# ---------------- Image Classification (X or IG Screenshot) ---------------- #
Ig_image = Image.open('sample_images/Screenshot 2568-05-27 at 16.09.52 copy.png')
X_image = Image.open('sample_images/Screenshot 2568-06-04 at 18.55.34.png')

@st.cache_resource
def load_resnet_model(model_type):
    os.makedirs("model", exist_ok=True)

    if model_type == "Detect X profile":
        model_path = "model/best_resnet18_model.pth"
        file_id = "1fJoH544qIINdmrO3KPTUb9TJQKntVZGE"
        download_from_drive(file_id, model_path)

        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        idx_to_class = {0: "Bot", 1: "Real", 2: "Cyborg"}

    elif model_type == "Detect Instagram profile":
        model_path = "model/best_resnet18_model_ig.pth"
        file_id = "15jb5frBeVffjp0NNCdHg2NXWvTTSDqpV"
        download_from_drive(file_id, model_path)

        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        idx_to_class = {0: "Bot", 1: "Real"}

    model.eval()
    return model, idx_to_class

# ------------- GradCAM and Inference Section ------------- #
def generate_gradcam_heatmap(model, input_tensor, target_class=None):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    handle_forward = model.layer4.register_forward_hook(forward_hook)
    handle_backward = model.layer4.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    if target_class is None:
        target_class = torch.argmax(output)
    class_score = output[0, target_class]
    class_score.backward()

    acts = activations['value'][0]
    grads = gradients['value'][0]
    weights = grads.mean(dim=[1, 2])
    cam = torch.zeros_like(acts[0])

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.cpu().numpy()

    cam = np.uint8(cam * 255)
    cam = Image.fromarray(cam).resize((224, 224), Image.BICUBIC)
    heatmap = np.array(cam)
    heatmap = plt.cm.jet(heatmap / 255.0)[:, :, :3]

    handle_forward.remove()
    handle_backward.remove()
    return heatmap

with st.expander("Fake X/IG profile detection"):
    st.markdown("#### Upload a screenshot of a profile (X or IG):")
    col1, col2 = st.columns(2)

    with col1:
        st.image(X_image, caption="Example of X screenshot", use_container_width=True)
    with col2:
        st.image(Ig_image, caption="Example of IG screenshot", use_container_width=True)

    st.warning("The Instagram profile detector is trained on a small dataset (60 samples) with limited variety. Predictions may not be fully accurate yet.")

    model_type = st.selectbox("Option", ["Detect X profile", "Detect Instagram profile"])
    uploaded_image = st.file_uploader("Upload your own screenshot", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Screenshot", use_container_width=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor = transform(img).unsqueeze(0)
        model, idx_to_class = load_resnet_model(model_type)

        with torch.no_grad():
            output = model(input_tensor)
            if model_type == "Detect Instagram profile" and output.shape[1] == 3:
                output[:, 2] = -float("inf")
            probs = torch.nn.functional.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            label = idx_to_class.get(pred, "Unknown")
            confidence = probs[0, pred].item()

        with torch.enable_grad():
            heatmap = generate_gradcam_heatmap(model, input_tensor, target_class=pred)
            img_resized = img.resize((224, 224))
            img_np = np.array(img_resized) / 255.0
            overlay = 0.4 * img_np + 0.6 * heatmap
            overlay = np.clip(overlay, 0, 1)

        st.success(f"**Predicted Class: {label} (Confidence: {confidence:.2%})**")
        st.markdown("#### Grad-CAM Heatmap")
        st.image(overlay, use_container_width=True)

    sample_dict = {
        "X Profile 01": "sample_images/res_profile_screenshot_1703800758487.png",
        "X Profile 02": "sample_images/res_profile_screenshot_1703802805126.png",
        "X Profile 03": "sample_images/res_profile_screenshot_1703806549494.png",
        "IG Profile 01": "sample_images/Screenshot 2568-05-26 at 15.55.17.png",
        "IG Profile 02": "sample_images/Screenshot 2568-05-26 at 15.55.26.png",
        "IG Profile 03": "sample_images/Screenshot 2568-05-27 at 17.03.27.png",
    }

    st.markdown("#### Choose a sample")
    sample_choice = st.selectbox("", list(sample_dict.keys()))
    sample_path = sample_dict[sample_choice]
    st.image(sample_path, caption=sample_choice, use_container_width=True)

    if st.button("Use this sample image"):
        img = Image.open(sample_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor = transform(img).unsqueeze(0)
        model, idx_to_class = load_resnet_model(model_type)

        with torch.no_grad():
            output = model(input_tensor)
            if model_type == "Detect Instagram profile" and output.shape[1] == 3:
                output[:, 2] = -float("inf")
            probs = torch.nn.functional.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            label = idx_to_class.get(pred, "Unknown")
            confidence = probs[0, pred].item()

        with torch.enable_grad():
            heatmap = generate_gradcam_heatmap(model, input_tensor, target_class=pred)
            img_resized = img.resize((224, 224))
            img_np = np.array(img_resized) / 255.0
            overlay = 0.4 * img_np + 0.6 * heatmap
            overlay = np.clip(overlay, 0, 1)

        st.success(f"**Predicted Class: {label} (Confidence: {confidence:.2%})**")
        st.markdown("#### Grad-CAM Heatmap")
        st.image(overlay, use_container_width=True)
