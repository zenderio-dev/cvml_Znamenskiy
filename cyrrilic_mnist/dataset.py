import io
import zipfile
from PIL import Image
import torch
from torch.utils.data import Dataset


class CyrillicDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform

        self.archive = zipfile.ZipFile(zip_path)

        self.images = []
        self.labels = []

        for file in self.archive.namelist():
            if not file.endswith(".png"):
                continue

            parts = [p for p in file.split("/") if p]

            if len(parts) >= 2:
                label = parts[-2]
                self.images.append(file)
                self.labels.append(label)

        self.classes = sorted(set(self.labels))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_bytes = self.archive.read(self.images[idx])

        image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

        # Накладываем на белый фон, чтобы убрать прозрачность
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)

        image = image.convert("L")

        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[self.labels[idx]]

        return image, torch.tensor(label, dtype=torch.long)
