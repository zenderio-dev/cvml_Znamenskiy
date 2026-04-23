import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import torch.optim as optim

save_path = Path(__file__).parent

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool1(x)
        x = self.tanh(self.conv2(x))
        x = self.pool2(x)
        x = self.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
    

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(), 
    transforms.Normalize((0.1307, ), (0.3081, ))
])
    
if __name__ == "__main__":
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model = LeNet5().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    import time
    t = time.perf_counter()
    for epoch in range(10):
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
        acc = 100.0 * correct / len(test_data)
        print(f"epoch {epoch+1}, acc={acc:.2f}%")
    torch.save(model.state_dict(), save_path / "lenet5.pth")
    print(f"elapsed time {time.perf_counter() - t:.2f} seconds")
