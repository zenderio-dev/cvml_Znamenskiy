import torch
import numpy as np
import matplotlib.pyplot as plt
from unet_road import RoadsDataset, load_model, DATA_PATH

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RoadsDataset(DATA_PATH, augment=False)
    model = load_model().to(device)

    image, mask = ds[0]

    with torch.no_grad():
        x = image.unsqueeze(0).to(device)
        pred = torch.sigmoid(model(x))
        pred = (pred > 0.5).float()

    true_mask = mask.squeeze().numpy()
    pred_mask = pred.squeeze().cpu().numpy()
    diff = np.abs(true_mask - pred_mask)

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(true_mask, cmap="gray")
    plt.title("Исходная маска")
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Предсказанная маска")
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(diff, cmap="gray")
    plt.title("Разница")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()