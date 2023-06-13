#! /bin/bash

set -e

echo "=== Generating SRS"
ezkl gen-srs --logrows 17 --params-path=ezkl/kzg.params

echo "=== Generating keys"
ezkl setup -M models/lenet.onnx --params-path=ezkl/kzg.params --vk-path=ezkl/vk.key --pk-path=ezkl/pk.key --circuit-params-path=ezkl/circuit.params

echo "=== Generating proof"
ezkl prove -M models/lenet.onnx -D models/input.json --pk-path=ezkl/pk.key --proof-path=ezkl/model.proof --params-path=ezkl/kzg.params --circuit-params-path=ezkl/circuit.params

echo "=== Verifying proof"
ezkl verify --proof-path=ezkl/model.proof --circuit-params-path=ezkl/circuit.params --vk-path=ezkl/vk.key --params-path=ezkl/kzg.params