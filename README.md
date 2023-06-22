# zkml-benchmark

Benchmarking EZKL.

## Setup

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Benchmarking

Use the `train_mnist.py` script to train a new model (which will show up in the `models` directory), e.g.:
```bash
python scripts/train_mnist.py --features 2 4 8
```

Next, you have to decide on some quantization parameters. A good start is to install `ezkl` >= 0.3.0 and run:
```bash
ezkl gen-settings -M models/lenet_2_4_8.onnx
ezkl calibrate-settings -M models/lenet_2_4_8.onnx -D models/input.json --target resources
```

This will generate a `settings.json` with suggested values for `scale`, `bits`, and `logrows`.
However, this is not always the fastest possible, so play with the numbers. Also, it was only calibrated on one training example.
In practice, you might need a higher value for `bits` and `logrows`.
The `measure_accuracy.py` script will print a warning if that is the case.

To measure the test set accuracy *after* quantization, run:
```bash
python scripts/measure_accuracy.py --model models/lenet_2_4_8.onnx --scale 4 --bits 15 --logrows 16
```

To generate a proof (and measure the time it took), run:

```bash
python scripts/generate_proof.py --model models/lenet_2_4_8.onnx --scale 4 --bits 15 --logrows 16
```
