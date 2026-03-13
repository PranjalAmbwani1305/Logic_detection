import torch
import torch.nn as nn

# Must match training values
NOISE_DIM = 100
NGF = 32          # use the same value you used in training
NUM_CHANNELS = 3  # RGB logos

class Generator(nn.Module):
    """
    DCGAN Generator: noise vector → fake logo image
    """
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # 100×1×1 → 256×4×4
            nn.ConvTranspose2d(NOISE_DIM, NGF*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF*8),
            nn.ReLU(True),

            # 256×4×4 → 128×8×8
            nn.ConvTranspose2d(NGF*8, NGF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*4),
            nn.ReLU(True),

            # 128×8×8 → 64×16×16
            nn.ConvTranspose2d(NGF*4, NGF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*2),
            nn.ReLU(True),

            # 64×16×16 → 32×32×32
            nn.ConvTranspose2d(NGF*2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),

            # 32×32×32 → 3×64×64
            nn.ConvTranspose2d(NGF, NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)