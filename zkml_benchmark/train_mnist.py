import pickle

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

from ezkl import export


class LeNet(nn.Module):
    def __init__(self, features=(6, 16, 120, 84)):
        super(LeNet, self).__init__()

        f1, f2, f3, f4 = features

        self.conv1 = nn.Conv2d(1, f1, kernel_size=5)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=5)
        self.fc1 = nn.Linear(f2 * 4 * 4, f3)
        if f4 is not None:
            self.fc2 = nn.Linear(f3, f4)
            self.fc3 = nn.Linear(f4, 10)
        else:
            self.fc2 = nn.Linear(f3, 10)
            self.fc3 = None

    def forward(self, x):
        # Note that tanh activation does a better job at keeping activations in a predictable
        # range. This means that fewer bits are needed for quantization!
        x = torch.tanh(self.conv1(x))
        x = torch.nn.functional.avg_pool2d(x, 2)
        x = torch.tanh(self.conv2(x))
        x = torch.nn.functional.avg_pool2d(x, 2)
        x = x.flatten(1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        if self.fc3 is not None:
            x = self.fc3(torch.tanh(x))
        return x


def load_dataset(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, train_dataset, learning_rate=0.001, num_epochs=10):

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the LeNet model
    total_step = len(train_dataset)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataset):

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')


def evaluate(model, test_dataset):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in test_dataset:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":

    train_dataset, test_dataset = load_dataset()

    model = LeNet((2, 4, 8, None))

    train(model, train_dataset)
    evaluate(model, test_dataset)

    with open("models/lenet.pickle", "wb") as f:
        pickle.dump(model, f)

    # Use first text example
    input_array = next(iter(test_dataset))[0][0].detach().numpy()

    export(model, input_array=input_array, input_shape=[
           1, 28, 28], onnx_filename="models/lenet.onnx", input_filename="models/input.json")
