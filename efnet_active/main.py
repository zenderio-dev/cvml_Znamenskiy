import torch
import cv2
import torch.nn as nn
import torchvision
from torchvision import transforms
import os
import time

MODEL_PATH = "model.pth"


def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(features, 1)
    return model


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(frame, model):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()

    label = "person" if prob > 0.5 else "no_person"
    return label, prob


def main():
    model = build_model()

    if not os.path.exists(MODEL_PATH):
        print("model.pth not found")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("Model loaded")

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

    print("p = predict, q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xff

        if key == ord("q"):
            break
        elif key == ord("p"):
            t = time.perf_counter()
            label, confidence = predict(frame, model)
            print(f"Elapsed time {time.perf_counter() - t:.4f}")
            print(label, confidence)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()