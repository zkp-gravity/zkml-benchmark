import pickle

from ezkl import export
from train_mnist import LeNet

if __name__ == "__main__":

    with open("models/lenet.pickle", "rb") as f:
        model = pickle.load(f)

    export(model, input_shape=[1, 28, 28], onnx_filename="models/lenet.onnx", input_filename="models/input.json")