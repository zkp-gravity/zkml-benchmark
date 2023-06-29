import argparse
import tempfile
import time

import ezkl

INPUTS_PATH = "models/input.json"


def timed_run(func):
    start = time.time()
    func()
    duration = time.time() - start
    print(f"  Took {duration:.2f}s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to the ONNX file', required=True)
    parser.add_argument('--scale', help='EZKL Scale argument', required=True)
    parser.add_argument('--bits', help='EZKL Bits argument', required=True)
    parser.add_argument(
        '--logrows', help='EZKL LogRows argument', required=True)
    args = parser.parse_args()

    model_path = args.model

    run_args = ezkl.PyRunArgs()
    run_args.scale = int(args.scale)
    run_args.bits = int(args.bits)
    run_args.logrows = int(args.logrows)

    with tempfile.TemporaryDirectory() as tmpdir:

        inputs_after_forward_path = f"{tmpdir}/input_after_forward.json"
        kzg_params_path = f"{tmpdir}/kzg_{run_args.logrows}.params"
        pk_path = f"{tmpdir}/pk.key"
        vk_path = f"{tmpdir}/vk.key"
        proof_path = f"{tmpdir}/lenet.proof"
        settings_path = f"{tmpdir}/settings.json"

        print("Generating Settings...")
        timed_run(lambda: ezkl.gen_settings(
            model_path, settings_path, run_args))

        print("Running forward pass...")
        timed_run(lambda: ezkl.gen_witness(INPUTS_PATH, model_path,
                                           inputs_after_forward_path, settings_path))

        print("Running mock...")
        timed_run(lambda: ezkl.mock(
            inputs_after_forward_path, model_path, settings_path))

        print("Generating KZG params...")
        timed_run(lambda: ezkl.gen_srs(kzg_params_path, run_args.logrows))

        print("Generating keys...")
        timed_run(lambda: ezkl.setup(model_path, vk_path, pk_path, kzg_params_path,
                                     settings_path))

        print("Generating proof...")
        timed_run(lambda: ezkl.prove(inputs_after_forward_path, model_path, pk_path, proof_path,
                                     kzg_params_path, "blake", "single", settings_path, False))

        print("Verifying proof...")
        timed_run(lambda: ezkl.verify(
            proof_path, settings_path, vk_path, kzg_params_path))
