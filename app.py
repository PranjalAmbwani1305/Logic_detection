import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# -----------------------------
# Hyperparameters
# -----------------------------
NOISE_DIM = 100
NGF = 32
NUM_CHANNELS = 3

# -----------------------------
# Generator Model
# -----------------------------
class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(

            nn.ConvTranspose2d(NOISE_DIM, NGF*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF*8, NGF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF*4, NGF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF*2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF, NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# -----------------------------
# Load Generator
# -----------------------------
generator = Generator()
generator.load_state_dict(torch.load("generator.pth", map_location="cpu"))
generator.eval()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎨 AI Logo Generator (DCGAN)")
st.write("Generate logos using a trained GAN model.")

# -----------------------------
# Generate Logos
# -----------------------------
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

# -----------------------------
# Upload Logo Section
# -----------------------------
st.subheader("Upload Logo")

uploaded = st.file_uploader("Upload logo image", type=["png","jpg","jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Logo")

    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0)

    # Generate sample logos
    noise = torch.randn(10, NOISE_DIM, 1, 1)

    with torch.no_grad():
        fake_images = generator(noise)

    similarities = []

    for fake in fake_images:
        fake = fake.unsqueeze(0)
        sim = F.cosine_similarity(
            img_tensor.flatten(),
            fake.flatten(),
            dim=0
        )
        similarities.append(sim.item())

    max_sim = max(similarities)

    if max_sim > 0.8:
        st.error("❌ Logo is NOT unique (similar pattern found)")
    else:
        st.success("✅ Logo appears UNIQUE")
