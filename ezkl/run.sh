#! /bin/bash

set -e

# Fails for a smaller number of bits
export BITS=18
export LOGROWS=19

echo "=== Running mock proof"
ezkl mock -M models/lenet.onnx -D models/input.json --bits $BITS --logrows $LOGROWS

if [ ! -f filename ]
then
    echo "=== Generating SRS"
    ezkl gen-srs --params-path=ezkl/kzg_$LOGROWS.params --logrows $LOGROWS
fi

echo "=== Generating keys"
ezkl setup -M models/lenet.onnx --params-path=ezkl/kzg_$LOGROWS.params --vk-path=ezkl/vk.key --pk-path=ezkl/pk.key --circuit-params-path=ezkl/circuit.params --bits $BITS --logrows $LOGROWS

echo "=== Generating proof"
ezkl prove -M models/lenet.onnx -D models/input.json --pk-path=ezkl/pk.key --proof-path=ezkl/model.proof --params-path=ezkl/kzg_$LOGROWS.params --circuit-params-path=ezkl/circuit.params

echo "=== Verifying proof"
ezkl verify --proof-path=ezkl/model.proof --circuit-params-path=ezkl/circuit.params --vk-path=ezkl/vk.key --params-path=ezkl/kzg_$LOGROWS.params