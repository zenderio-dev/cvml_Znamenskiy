import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["square", "circle", "triangle"]

curpath = Path(__file__).resolve().parent
data_root = curpath / "shapes"


class ShapesDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = []

        for cls_name in classes:
            img_dir = root / cls_name / "images"
            meta_dir = root / cls_name / "labels"

            for img_path in sorted(img_dir.glob("*.png")):
                labels_path = meta_dir / (img_path.stem + ".txt")
                if labels_path.exists():
                    self.images.append((img_path, labels_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, labels_path = self.images[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            tensor = self.transform(Image.fromarray(img))
        else:
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        cls, x, y, w, h = map(float, labels_path.read_text().split())
        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)

        return tensor, int(cls), bbox


class SimpleDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(2),
        )

        self.cls_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.bbox_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        self.cls_head = nn.Linear(128, num_classes)
        self.bbox_head = nn.Linear(256, 4)

    def forward(self, x):
        features = self.backbone(x)
        cls_features = self.cls_branch(features)
        bbox_features = self.bbox_branch(features)
        cls_pred = self.cls_head(cls_features)
        bbox_pred = torch.sigmoid(self.bbox_head(bbox_features))
        return cls_pred, bbox_pred


def xywh_to_xyxy(box):
    x1 = box[:, 0] - box[:, 2] / 2
    y1 = box[:, 1] - box[:, 3] / 2
    x2 = box[:, 0] + box[:, 2] / 2
    y2 = box[:, 1] + box[:, 3] / 2
    return x1, y1, x2, y2


def giou_loss(pred, target):
    p_x1, p_y1, p_x2, p_y2 = xywh_to_xyxy(pred)
    t_x1, t_y1, t_x2, t_y2 = xywh_to_xyxy(target)

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    area_t = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)
    union = area_p + area_t - inter + 1e-7
    iou = inter / union

    c_x1 = torch.min(p_x1, t_x1)
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2)
    c_y2 = torch.max(p_y2, t_y2)
    area_c = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0) + 1e-7

    giou = iou - (area_c - union) / area_c
    return (1 - giou).mean()


def compute_iou(pred, target):
    p_x1, p_y1, p_x2, p_y2 = xywh_to_xyxy(pred)
    t_x1, t_y1, t_x2, t_y2 = xywh_to_xyxy(target)

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    area_t = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)
    union = area_p + area_t - inter + 1e-7

    return inter / union


def detection_loss(cls_pred, bbox_pred, cls_targets, bbox_targets, lambda_bbox=30.0):
    loss_cls = F.cross_entropy(cls_pred, cls_targets)
    loss_bbox = F.smooth_l1_loss(bbox_pred, bbox_targets) + giou_loss(bbox_pred, bbox_targets)
    return loss_cls + lambda_bbox * loss_bbox, loss_cls, loss_bbox


transform = transforms.Compose([
    transforms.ToTensor(),
])


def evaluate(model, loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    mean_iou = 0.0

    with torch.no_grad():
        for images, cls_t, bbox_t in loader:
            images = images.to(device)
            cls_t = cls_t.to(device)
            bbox_t = bbox_t.to(device)

            cls_pred, bbox_pred = model(images)
            loss, _, _ = detection_loss(cls_pred, bbox_pred, cls_t, bbox_t)

            val_loss += loss.item()
            correct += (cls_pred.argmax(1) == cls_t).sum().item()
            total += cls_t.size(0)
            mean_iou += compute_iou(bbox_pred, bbox_t).mean().item()

    return val_loss / len(loader), correct / total, mean_iou / len(loader)


def train_dataset(dataset_name, epochs=30):
    root = data_root / dataset_name

    train_ds = ShapesDataset(root / "train", transform=transform)
    val_ds = ShapesDataset(root / "val", transform=transform)
    test_ds = ShapesDataset(root / "test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = SimpleDetector(num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    save_path = root / "best.pt"
    history = defaultdict(list)
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_cls = train_box = 0.0

        for images, cls_t, bbox_t in train_loader:
            images = images.to(device)
            cls_t = cls_t.to(device)
            bbox_t = bbox_t.to(device)

            optimizer.zero_grad()
            cls_pred, bbox_pred = model(images)
            loss, lc, lb = detection_loss(cls_pred, bbox_pred, cls_t, bbox_t)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_cls += lc.item()
            train_box += lb.item()

        val_loss, val_acc, val_iou = evaluate(model, val_loader)

        history["train_loss"].append(train_loss / len(train_loader))
        history["train_cls"].append(train_cls / len(train_loader))
        history["train_box"].append(train_box / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_iou"].append(val_iou)

        scheduler.step()

        score = val_acc + val_iou
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), save_path)

        print(
            f"{dataset_name} | epoch {epoch:02d}/{epochs} | "
            f"val_loss={val_loss:.4f} | acc={val_acc:.4f} | iou={val_iou:.4f}"
        )

    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc, test_iou = evaluate(model, test_loader)

    return model, test_loader, history, {
        "dataset": dataset_name,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_iou": test_iou,
    }


def show_predictions(loader, model, save_name, n=8):
    model.eval()
    images, cls_t, bbox_t = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        cls_pred, bbox_pred = model(images)

    preds = cls_pred.argmax(1).cpu()

    fig, axes = plt.subplots(2, n // 2, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
        h, w = img_np.shape[:2]

        cx, cy, bw, bh = bbox_t[i].numpy()
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        ax.add_patch(
            Rectangle(
                (x1, y1),
                bw * w,
                bh * h,
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
            )
        )

        cx, cy, bw, bh = bbox_pred[i].cpu().numpy()
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        ax.add_patch(
            Rectangle(
                (x1, y1),
                bw * w,
                bh * h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
        )

        gt_name = classes[cls_t[i]]
        pr_name = classes[preds[i]]
        color = "green" if preds[i] == cls_t[i] else "red"
        ax.set_title(f"GT:{gt_name}  Pred:{pr_name}", color=color, fontsize=9)
        ax.imshow(img_np)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(curpath / save_name, dpi=150, bbox_inches="tight")
    plt.close()


def plot_history(history, save_name, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.title(f"{title}: loss")
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(f"{title}: metrics")
    plt.plot(history["val_acc"], label="acc")
    plt.plot(history["val_iou"], label="iou")
    plt.legend()

    plt.tight_layout()
    plt.savefig(curpath / save_name, dpi=150, bbox_inches="tight")
    plt.close()


epochs = 30
results = []

model, test_loader, history, metrics = train_dataset("shapes_dataset", epochs=epochs)
results.append(metrics)

plot_history(history, "history_shapes_dataset.png", "shapes_dataset")
show_predictions(test_loader, model, "predictions_shapes_dataset.png")

for dataset_name in ["shapes_dataset_bg", "shapes_dataset_random"]:
    _, _, _, metrics = train_dataset(dataset_name, epochs=epochs)
    results.append(metrics)

print("\nИтог:")
for item in results:
    print(
        f"{item['dataset']}: "
        f"test_acc={item['test_acc']:.4f}, "
        f"test_iou={item['test_iou']:.4f}"
    )
