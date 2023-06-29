# zkml-benchmark

Benchmarking EZKL.

## Setup

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Benchmarking EZKL

Use the `ezkl_train_mnist.py` script to train a new model (which will show up in the `models` directory), e.g.:
```bash
python scripts/ezkl_train_mnist.py --features 2 4 8
```

Next, you have to decide on some quantization parameters. A good start is to install [`ezkl`](https://github.com/zkonduit/ezkl) >= 0.3.0 and run:
```bash
ezkl gen-settings -M models/lenet_2_4_8.onnx
ezkl calibrate-settings -M models/lenet_2_4_8.onnx -D models/input.json --target resources
```

This will generate a `settings.json` with suggested values for `scale`, `bits`, and `logrows`.
However, this is not always the fastest possible, so play with the numbers. Also, it was only calibrated on one training example.
In practice, you might need a higher value for `bits` and `logrows`.
The `ezkl_measure_accuracy.py` script will print a warning if that is the case.

To measure the test set accuracy *after* quantization, run:
```bash
python scripts/ezkl_measure_accuracy.py --model models/lenet_2_4_8.onnx --scale 4 --bits 15 --logrows 16
```

To generate a proof (and measure the time it took), run:

```bash
python scripts/ezkl_generate_proof.py --model models/lenet_2_4_8.onnx --scale 4 --bits 15 --logrows 16
```

## Benchmarking Daniel Kang's `zkml`

Clone [ddkang/zkml](https://github.com/ddkang/zkml) next to this repository and follow the instruction in the readme until you get their basic example working. This will also give you the proving time for their checked-in MNIST model.

Then, the `scripts/zkml_measure_accuracy.py` script will measure accuracy of the checked-in on the  MNIST test set (by repeadedly running their `./target/release/test_circuit` command-line tool).