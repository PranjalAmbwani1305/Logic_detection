import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# --------------------
# Hyperparameters
# --------------------

NOISE_DIM = 100
NGF = 32
NDF = 32
NUM_CHANNELS = 3

device = torch.device("cpu")

# --------------------
# Generator
# --------------------

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

# --------------------
# Discriminator
# --------------------

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(

            nn.Conv2d(NUM_CHANNELS, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF, NDF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF*2, NDF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF*4, NDF*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# --------------------
# Load Models
# --------------------

generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load("discriminator.pth", map_location=device))
discriminator.eval()

# --------------------
# Streamlit UI
# --------------------

st.title("🎨 AI Logo Generator & Detector")

st.write("Generate logos using DCGAN and check uploaded logos.")

# --------------------
# Generate Logos
# --------------------

num = st.slider("Number of logos",1,8,4)

if st.button("Generate Logos"):

    noise = torch.randn(num, NOISE_DIM, 1, 1).to(device)

    with torch.no_grad():
        fake = generator(noise).cpu()

    cols = st.columns(num)

    for i in range(num):

        img = (fake[i].permute(1,2,0)+1)/2
        cols[i].image(img.numpy(), caption=f"Logo {i+1}")

# --------------------
# Upload Logo
# --------------------

st.subheader("Upload Logo")

uploaded = st.file_uploader("Upload logo image", type=["png","jpg","jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Logo")

    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = discriminator(img).item()

    st.write("Prediction Score:", round(pred,3))

    if pred > 0.5:
        st.success("✅ Logo looks REAL (similar to training logos)")
    else:
        st.error("❌ Logo looks FAKE / UNIQUE")
