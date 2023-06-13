from run_args import run_args

import ezkl

MODEL_PATH = "models/lenet.onnx"
INPUTS_PATH = "models/input.json"
KZG_PARAMS_PATH = f"ezkl/kzg_{run_args.logrows}.params"
CIRCUIT_PARAMS_PATH = "ezkl/circuit.params"
PK_PATH = "ezkl/pk.key"
VK_PATH = "ezkl/vk.key"
PROOF_PATH = "ezkl/lenet.proof"

if __name__ == "__main__":

    print("Running mock...")
    ezkl.mock(INPUTS_PATH, MODEL_PATH, run_args)

    print("Generating KZG params...")
    ezkl.gen_srs(KZG_PARAMS_PATH, run_args.logrows)

    print("Generating keys...")
    ezkl.setup(MODEL_PATH, VK_PATH, PK_PATH, KZG_PARAMS_PATH, CIRCUIT_PARAMS_PATH, run_args)

    print("Generating proof...")
    ezkl.prove(INPUTS_PATH, MODEL_PATH, PK_PATH, PROOF_PATH, KZG_PARAMS_PATH, "blake", "single", CIRCUIT_PARAMS_PATH)

    print("Verifying proof...")
    ezkl.verify(PROOF_PATH, CIRCUIT_PARAMS_PATH, VK_PATH, KZG_PARAMS_PATH)