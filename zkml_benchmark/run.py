import time

from run_args import run_args

import ezkl

MODEL_PATH = "models/lenet.onnx"
INPUTS_PATH = "models/input.json"
INPUTS_AFTER_FORWARD_PATH = "models/input_after_forward.json"
KZG_PARAMS_PATH = f"ezkl/kzg_{run_args.logrows}.params"
CIRCUIT_PARAMS_PATH = "ezkl/circuit.params"
PK_PATH = "ezkl/pk.key"
VK_PATH = "ezkl/vk.key"
PROOF_PATH = "ezkl/lenet.proof"


def timed_run(func):
    start = time.time()
    func()
    duration = time.time() - start
    print(f"  Took {duration:.2f}s")


if __name__ == "__main__":

    print("Running forward pass...")
    timed_run(lambda: ezkl.forward(INPUTS_PATH, MODEL_PATH,
              INPUTS_AFTER_FORWARD_PATH, run_args))

    print("Running mock...")
    timed_run(lambda: ezkl.mock(
        INPUTS_AFTER_FORWARD_PATH, MODEL_PATH, run_args))

    print("Generating KZG params...")
    timed_run(lambda: ezkl.gen_srs(KZG_PARAMS_PATH, run_args.logrows))

    print("Generating keys...")
    timed_run(lambda: ezkl.setup(MODEL_PATH, VK_PATH, PK_PATH, KZG_PARAMS_PATH,
                                 CIRCUIT_PARAMS_PATH, run_args))

    print("Generating proof...")
    timed_run(lambda: ezkl.prove(INPUTS_AFTER_FORWARD_PATH, MODEL_PATH, PK_PATH, PROOF_PATH,
                                 KZG_PARAMS_PATH, "blake", "single", CIRCUIT_PARAMS_PATH))

    print("Verifying proof...")
    timed_run(lambda: ezkl.verify(
        PROOF_PATH, CIRCUIT_PARAMS_PATH, VK_PATH, KZG_PARAMS_PATH))
