import os
import re
import subprocess
import tempfile

import msgpack
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


def print_msgpack_input(input_path):

    with open(input_path, "rb") as f:
        inp = f.read()
    data = msgpack.unpackb(inp, raw=False)
    print(data)

    img = np.array(data[0]["data"]).reshape(
        data[0]["shape"])[0, :, :, 0]
    print("Image is in range {} to {}".format(img.min(), img.max()))
    for row in range(28):
        print("".join("X" if img[row, col] > 0
              else " " for col in range(28)))


def predict(image):

    with tempfile.TemporaryDirectory() as tmpdir:

        image = (image * 1024).numpy().astype(np.int32)

        input_path = os.path.join(tmpdir, "input.msgpack")
        with open(input_path, "wb") as f:
            data = [{"idx": 0, "shape": (
                1, 28, 28, 1), "data": image.flatten().tolist()}]
            f.write(msgpack.packb(
                data, use_bin_type=True))

        result = subprocess.run(["./target/release/test_circuit", "examples/mnist/model.msgpack",
                                input_path, "kzg"], stdout=subprocess.PIPE)
        pattern = re.compile(r"final out\[(\d)\] x: (-?\d+)")

        scores = []
        for line in result.stdout.decode("utf-8").split("\n")[-30:]:
            match = pattern.match(line)
            if match:
                index = int(match.group(1))
                score = int(match.group(2))

                assert index == len(scores)
                scores.append(score)

        assert len(scores) == 10
        prediction = np.argmax(scores)
        return prediction


def compute_accuracy():

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False)

    total = 0
    correct = 0

    for image, labels in tqdm(test_loader):

        prediction = predict(image)

        total += 1
        if prediction == labels.item():
            correct += 1

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":

    os.chdir("../zkml")
    compute_accuracy()
