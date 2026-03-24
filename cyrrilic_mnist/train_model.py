import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from dataset import CyrillicDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())

    # transform
    train_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # dataset
    full_dataset = CyrillicDataset("Cyrillic.zip", transform=None)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_subset, test_subset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # transform
    train_subset.dataset.transform = train_transform

    test_dataset = CyrillicDataset("Cyrillic.zip", transform=test_transform)
    test_subset.dataset = test_dataset

    # loader
    train_loader = DataLoader(
        train_subset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    classes = full_dataset.classes
    num_classes = len(classes)

    model = CNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()

        train_running_loss = 0.0
        train_total = 0
        train_correct = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]")

        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_running_loss / train_total
        train_acc = train_correct / train_total

        # test
        model.eval()

        test_running_loss = 0.0
        test_total = 0
        test_correct = 0

        with torch.no_grad():
            test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [test]")

            for images, labels in test_bar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_running_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss = test_running_loss / test_total
        test_acc = test_correct / test_total

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        # сохраняем лучшую модель
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "test_acc": best_test_acc
                },
                "model.pth"
            )

    print(f"Best test accuracy: {best_test_acc:.4f}")

    # графики
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="train_loss")
    plt.plot(test_loss_history, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label="train_acc")
    plt.plot(test_acc_history, label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("train.png")
    plt.close()


if __name__ == "__main__":
    main()
