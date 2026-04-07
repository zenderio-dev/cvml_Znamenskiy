import torch
import cv2
import torch.nn as nn
import torchvision
from torchvision import transforms
from collections import deque
import os

MODEL_PATH = "model.pth"


def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(features, 1)
    return model


class Buffer:
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)

    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)

    def get_batch(self):
        images = torch.stack(list(self.frames))
        labels = torch.tensor(list(self.labels), dtype=torch.float32)
        return images, labels


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def train(buffer, model, criterion, optimizer):
    if len(buffer) < 10:
        return None

    model.train()
    images, labels = buffer.get_batch()

    optimizer.zero_grad()
    predictions = model(images).squeeze(1)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    model = build_model()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print("Model loaded")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

    buffer = Buffer()
    count_labeled = 0

    print("1 = person, 2 = no_person, s = save, q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xff
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if key == ord("q"):
            break
        elif key == ord("1"):
            tensor = transform(image)
            buffer.append(tensor, 1.0)
            count_labeled += 1
            print(f"person added [{len(buffer)}/{buffer.frames.maxlen}]")
        elif key == ord("2"):
            tensor = transform(image)
            buffer.append(tensor, 0.0)
            count_labeled += 1
            print(f"no_person added [{len(buffer)}/{buffer.frames.maxlen}]")
        elif key == ord("s"):
            torch.save(model.state_dict(), MODEL_PATH)
            print("Model saved")

        if count_labeled >= buffer.frames.maxlen:
            loss = train(buffer, model, criterion, optimizer)
            if loss is not None:
                print(f"Loss = {loss:.6f}")
            count_labeled = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()