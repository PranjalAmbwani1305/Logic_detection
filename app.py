import streamlit as st
import torch
import numpy as np
from PIL import Image
from generator_model import Generator

NOISE_DIM = 100

st.set_page_config(page_title="AI Logo Generator", layout="wide")

st.title("🎨 AI Logo Generator (DCGAN)")
st.write("Generate logos using a trained GAN model.")

# Load model
generator = Generator()
generator.load_state_dict(torch.load("generator.pth", map_location="cpu"))
generator.eval()

# Slider
num_images = st.slider("Number of logos", 1, 12, 4)

if st.button("Generate Logos"):

    noise = torch.randn(num_images, NOISE_DIM, 1, 1)

    with torch.no_grad():
        fake_images = generator(noise).cpu()

    cols = st.columns(num_images)

    for i in range(num_images):

        img = (fake_images[i].permute(1,2,0)+1)/2
        img = img.numpy()

        cols[i].image(img, caption=f"Logo {i+1}")

st.divider()

# Upload logo section
st.subheader("Upload Logo")

uploaded = st.file_uploader("Upload a logo image")

if uploaded:

    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image")
    st.success("Logo uploaded successfully")