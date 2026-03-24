import torch
from PIL import Image
from torchvision import transforms

from train_model import CNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("model.pth", map_location=device)

    classes = checkpoint["classes"]

    model = CNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open("test.png").convert("RGBA")

    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(background, img)

    img = img.convert("L")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()

    print("Predicted letter:", classes[pred])


if __name__ == "__main__":
    main()
