import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler


def weight_initialization(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class StandardResidualBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, dropout=0.5):
        super(StandardResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Identity()

        self.apply(weight_initialization)


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.batch_norm1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class CoefficientNetwork(nn.Module):
    def __init__(self, d_hidden, dropout=0.5):
        super(CoefficientNetwork, self).__init__()
        self.fc1 = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.apply(weight_initialization)


    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x_flat = torch.flatten(x, 1)
        
        coeff = F.relu(self.fc1(x_flat))
        coeff = self.dropout(coeff)
        coeff = self.fc2(coeff)
        coeff = coeff.view(batch_size, channels, height, width)
        coeff = torch.sigmoid(coeff)
        return coeff


class DynamicResidualBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, dropout=0.5):
        super(DynamicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Identity()
        self.coeff_net = CoefficientNetwork(input_channels * 32 * 32, dropout)

        self.apply(weight_initialization)


    def forward(self, x):
        coeff = self.coeff_net(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.batch_norm1(out)
        out = self.dropout(out)
        out = self.conv2(out)

        residual = self.shortcut(x)
        out += coeff * residual
        return out


class DynCIFAR10Model(nn.Module):
    def __init__(self, input_channels=3, d_hidden=256, dropout=0.5):
        super(DynCIFAR10Model, self).__init__()

        self.res_block1 = DynamicResidualBlock(input_channels, d_hidden, dropout)
        self.res_block2 = DynamicResidualBlock(input_channels, d_hidden, dropout)
        self.res_block3 = DynamicResidualBlock(input_channels, d_hidden, dropout)

        self.fc1 = nn.Linear(input_channels * 32 * 32, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, 10)

        self.apply(weight_initialization)


    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class StandardCIFAR10Model(nn.Module):
    def __init__(self, input_channels=3, d_hidden=256, dropout=0.5):
        super(StandardCIFAR10Model, self).__init__()

        self.res_block1 = StandardResidualBlock(input_channels, d_hidden, dropout)
        self.res_block2 = StandardResidualBlock(input_channels, d_hidden, dropout)
        self.res_block3 = StandardResidualBlock(input_channels, d_hidden, dropout)

        self.fc1 = nn.Linear(input_channels * 32 * 32, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, 10)

        self.apply(weight_initialization)


    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train_model_with_scheduler(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs=50, accumulation_steps=1, early_stopping_patience=-1):
    scaler = GradScaler()

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    best_test_accuracy = 0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = compute_accuracy(model, train_loader, device)
        test_accuracy = compute_accuracy(model, test_loader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        scheduler.step(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

        if early_stopping_patience != -1:
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    return train_losses, train_accuracies, test_accuracies


def load_data(batch_size=128):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)

    return train_loader, test_loader


def plot_metrics(metrics_standard, metrics_dynamic):
    epochs = range(1, len(metrics_standard[0]) + 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, metrics_standard[0], label="Standard - Loss")
    plt.plot(epochs, metrics_dynamic[0], label="Dynamic - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, metrics_standard[1], label="Standard - Train Acc")
    plt.plot(epochs, metrics_dynamic[1], label="Dynamic - Train Acc")
    plt.plot(epochs, metrics_standard[2], label="Standard - Test Acc")
    plt.plot(epochs, metrics_dynamic[2], label="Dynamic - Test Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("figureDynConv.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 0.001
    batch_size = 512
    num_epochs = 100

    train_loader, test_loader = load_data(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()

    dynamic_model = DynCIFAR10Model(input_channels=3, d_hidden=512).to(device)
    optimizer = optim.AdamW(dynamic_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

    print("Training Dynamic Model...")
    metrics_dynamic = train_model_with_scheduler(dynamic_model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs)

    standard_model = StandardCIFAR10Model(input_channels=3, d_hidden=512).to(device)
    optimizer = optim.AdamW(standard_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

    print("Training Standard Model...")
    metrics_standard = train_model_with_scheduler(standard_model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs)

    plot_metrics(metrics_standard, metrics_dynamic)