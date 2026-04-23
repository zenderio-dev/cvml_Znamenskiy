import json
import os
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms(model_name):
    if model_name == "b0":
        image_size = 224
    elif model_name == "b1":
        image_size = 240
    elif model_name == "b2":
        image_size = 260
    else:
        raise ValueError("Unsupported model")

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, val_transform


def get_model(model_name, num_classes):
    if model_name == "b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights)
    elif model_name == "b1":
        weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
        model = torchvision.models.efficientnet_b1(weights=weights)
    elif model_name == "b2":
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights)
    else:
        raise ValueError("Unsupported model")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def get_dataloaders(data_root, model_name, batch_size, num_workers):
    train_transform, val_transform = get_transforms(model_name)

    train_dir = os.path.join(data_root, "images", "train")
    val_dir = os.path.join(data_root, "images", "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, train_dataset.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    return total_loss / total, correct / total, y_true, y_pred


def save_confusion_matrix(y_true, y_pred, class_names, output_path, title):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_model(model_name, data_root, output_dir, epochs=10, batch_size=16, lr=0.0003, num_workers=0):
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    train_loader, val_loader, class_names = get_dataloaders(
        data_root,
        model_name,
        batch_size,
        num_workers
    )

    model = get_model(model_name, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"{model_name.upper()} | "
            f"epoch {epoch + 1}/{epochs} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    weights_path = os.path.join(output_dir, f"efficientnet_{model_name}.pth")
    torch.save(model.state_dict(), weights_path)

    _, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)

    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        cm_path,
        f"Confusion Matrix - EfficientNet-{model_name.upper()}"
    )

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        ),
        "history": history
    }

    metrics_path = os.path.join(output_dir, f"metrics_{model_name}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    return {
        "model": model_name,
        "accuracy": val_acc,
        "weights_path": weights_path,
        "confusion_matrix_path": cm_path,
        "metrics_path": metrics_path
    }