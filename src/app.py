import os
import gdown
import numpy as np
import cv2 as cv
import streamlit as st

# ===== Load model and cluster centers once =====
@st.cache_resource
def load_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("resources", exist_ok=True)

    model_path = "models/colorization_release_v2.caffemodel"
    proto_path = "models/colorization_deploy_v2.prototxt"
    pts_path = "resources/pts_in_hull.npy"

    # ✅ Download model if missing
    if not os.path.exists(model_path):
        st.warning("Downloading model... please wait ⏳ (~130MB)")
        
        url = "https://drive.google.com/uc?id=1jSi0P1hc7RJsv3Zj-4fAQZT1JqMil3-6"  # replace if needed
        gdown.download(url, model_path, quiet=False)

    # ✅ Load pts
    if not os.path.exists(pts_path):
        st.error("Missing pts_in_hull.npy file")
        st.stop()

    pts_in_hull = np.load(pts_path)
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)

    # ✅ Load model
    net = cv.dnn.readNetFromCaffe(proto_path, model_path)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    return net

# ===== Image colorization function =====
def colorize_image(image, net):
    # Convert PIL to OpenCV format
    image_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    image_cv = image_cv.astype(np.float32) / 255.0
    h, w = image_cv.shape[:2]

    # Convert to Lab and extract L channel
    lab = cv.cvtColor(image_cv, cv.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0]

    # Resize and subtract 50
    l_rs = cv.resize(l_channel, (224, 224)) - 50
    net.setInput(cv.dnn.blobFromImage(l_rs))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize and merge with original L
    ab_dec_us = cv.resize(ab_dec, (w, h))
    lab_out = np.concatenate((l_channel[:, :, np.newaxis], ab_dec_us), axis=2)
    bgr_out = cv.cvtColor(lab_out, cv.COLOR_Lab2BGR)
    bgr_out = np.clip(bgr_out, 0, 1)

    return (bgr_out * 255).astype(np.uint8)

# ===== Streamlit UI =====
st.set_page_config(page_title="B&W Image Colorization", layout="centered")
st.title("🖼️ Black & White Image Colorizer")

# 👇 ADD HERE
st.warning("First run may take time as model downloads (~130MB)")

st.write("Upload a grayscale image and convert it into a colorized version using Deep Learning.")

# Load model
model = load_model()

# Load model
model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload a B&W image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("🎨 Colorize"):
        with st.spinner("Colorizing..."):
            result = colorize_image(image, model)
            result_img = Image.fromarray(cv.cvtColor(result, cv.COLOR_BGR2RGB))
            st.image(result_img, caption="Colorized Image", use_column_width=True)

            # Download button
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                result_img.save(tmp_file.name)
                st.download_button(
                    label="📥 Download Colorized Image",
                    data=open(tmp_file.name, "rb").read(),
                    file_name="colorized_result.png",
                    mime="image/png"
                )
