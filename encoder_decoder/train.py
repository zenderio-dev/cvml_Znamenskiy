import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
from torch import nn
import torch.optim as optim
import random
import string
from pathlib import Path

letters = string.ascii_letters + string.digits

class ImageDataset(Dataset):
    def __init__(self, n_samples=200, img_size=128, variant=1):
        super().__init__()
        self.n = n_samples
        self.size = img_size
        self.variant = variant
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = Image.new('L', (self.size, self.size), color=255)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        text = ""
        x = 0
        y = 0

        if self.variant == 1:
            text = "ABCDE"
            x = np.random.randint(5, self.size - 50)
            y = np.random.randint(5, self.size - 50)
        elif self.variant == 2:
            text_len = 4
            text = ''.join(random.choice(letters) for _ in range(text_len))
            x = 25
            y = 25
        elif self.variant == 3:
            text_len = random.randint(4, 10)
            text = ''.join(random.choice(letters) for _ in range(text_len))
            x = 25
            y = 25
        elif self.variant == 4:
            text_len = random.randint(4, 10)
            text = ''.join(random.choice(letters) for _ in range(text_len))
            x = np.random.randint(5, self.size - 50)
            y = np.random.randint(5, self.size - 50)

        draw.text((x, y), text, fill=0, font=font)
        tensor = self.transform(img)
        return tensor, tensor

class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.bottleneck = nn.Linear(256*16*16, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.bottleneck = nn.Linear(latent_dim, 256*16*16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    for variant in range(1, 5):
        dataset = ImageDataset(n_samples=2000, img_size=256, variant=variant)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        encoder = Encoder()
        decoder = Decoder()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder.to(device)
        decoder.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

        epochs = 10
        for epoch in range(epochs):
            running_loss = 0.0
            for imgs, _ in dataloader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                latent = encoder(imgs)
                output = decoder(latent)
                loss = criterion(imgs, output)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(dataloader)
            print(f"[Variant {variant}] Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        torch.save(encoder.state_dict(), BASE_DIR / f"encoder_{variant}.pth")
        torch.save(decoder.state_dict(), BASE_DIR / f"decoder_{variant}.pth")