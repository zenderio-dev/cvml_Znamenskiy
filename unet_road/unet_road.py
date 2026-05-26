import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np

DATA_PATH = Path("roads")
MODEL_PATH = Path("unet_road_model.pth")

class RoadsDataset(Dataset):
    def __init__(self, path, size=(256, 256), augment=True):
        self.images = sorted((path / "images").glob("*.png"))
        self.masks = sorted((path / "masks").glob("*.png"))
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert("RGB").resize(self.size)
        mask = Image.open(self.masks[i]).convert("L").resize(self.size)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)
        mask = (mask == 82).astype(np.float32)[None, :, :]

        if self.augment and np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=2).copy()

        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).float()
        return img, mask

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for f in features:
            self.down.append(DoubleConv(in_ch, f))
            in_ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.up.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.up.append(DoubleConv(f * 2, f))

        self.out = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []

        for layer in self.down:
            x = layer(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.up), 2):
            x = self.up[i](x)
            x = torch.cat((skips[i // 2], x), dim=1)
            x = self.up[i + 1](x)

        return self.out(x)

class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred).reshape(-1)
        target = target.reshape(-1)
        inter = (pred * target).sum()
        return 1 - (2 * inter + 1) / (pred.sum() + target.sum() + 1)

def train(epochs=20, batch_size=4, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RoadsDataset(DATA_PATH)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            out = model(images)
            loss = loss_fn(out, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                pred = (torch.sigmoid(out) > 0.5).float()
                correct += (pred == masks).sum().item()
                total += masks.numel()

        acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, loss={total_loss / len(loader):.4f}, acc={acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    train()