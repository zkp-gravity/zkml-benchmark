import json
import os
import pickle
import tempfile

import torch
import torchvision
import torchvision.transforms as transforms
from run_args import run_args
# Needed to load pickle file
from train_mnist import LeNet

import ezkl


def predict(x, model, tmpdir):

    input_filename = os.path.join(tmpdir, "input.json")
    output_filename = os.path.join(tmpdir, "output.json")

    torch_out = model(x)
    _, predicted = torch.max(torch_out, 1)

    data_array = ((x).detach().numpy()).reshape([-1]).tolist()
    data = dict(input_shapes=[x.shape[1:]] * x.shape[0],
                input_data=[data_array],
                output_data=[((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

    with open(input_filename, "w") as f:
        json.dump(data, f)

    ezkl.forward(input_filename, "models/lenet.onnx",
                 output_filename, run_args)

    with open(output_filename, "r") as f:
        scores = json.load(f)["output_data"]
    _, predicted_quantized = torch.max(torch.tensor(scores), 1)

    return predicted, predicted_quantized


def compute_accuracy(model, tmpdir):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform)
    # Only works with batch size 1?
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False)

    total = 0
    correct = 0
    correct_quantized = 0

    for i, (images, labels) in enumerate(test_loader):
        predicted, predicted_quantized = predict(images, model, tmpdir)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        correct_quantized += (predicted_quantized == labels).sum().item()

        if (i + 1) % 1000 == 0:
            print(f"Example {i + 1} of {len(test_dataset)}")

    print(f"Measured accuracy on {total} samples")

    accuracy = 100 * correct / total
    accuracy_quantized = 100 * correct_quantized / total
    print(
        f'Test Accuracy: {accuracy:.2f}% (original), {accuracy_quantized:.2f}% (quantized)')


if __name__ == "__main__":

    with open("models/lenet.pickle", "rb") as f:
        model = pickle.load(f)

    with torch.no_grad():
        with tempfile.TemporaryDirectory() as tempdir:

            compute_accuracy(model, tempdir)