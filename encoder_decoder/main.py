import torch
import matplotlib.pyplot as plt
from train import Encoder, Decoder, ImageDataset

encoder = Encoder()
decoder = Decoder()

encoder.load_state_dict(torch.load('encoder_4.pth'))
decoder.load_state_dict(torch.load('decoder_4.pth'))
encoder.eval()
decoder.eval()

dataset = ImageDataset(10, 256, variant=4)
image, _ = dataset[0]

with torch.no_grad():
    latent = encoder(image.unsqueeze(0))
    result = decoder(latent)

    plt.subplot(131)
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Original')

    plt.subplot(132)
    plt.imshow(result.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Reconstructed')

    plt.subplot(133)
    plt.imshow(image.squeeze().cpu().numpy() - result.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Difference')

    plt.show()