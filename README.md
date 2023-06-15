# zkml-benchmark
Benchmarking different ZKML approaches

## Setup

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Benchmarking


The following script trains a new MNIST model, writing `models/lenet.pickle`, `models/lenet.onnx`, and `models/input.json`:

```
python zkml_benchmark/train_mnist.py
```

Head to [`zkml_benchmark/run_args.py`](zkml_benchmark/run_args.py) to set the quantization parameters.

The following script measures the accuracy of the model (both quantized and not quantized) on the MNIST test set:
```
python zkml_benchmark/measure_accuracy.py
```

The following script runs the proving pipeline:
```
python zkml_benchmark/run.py
```