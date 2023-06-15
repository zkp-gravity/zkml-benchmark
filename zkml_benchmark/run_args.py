import ezkl

# Output of:
# ezkl calibrate-settings -M models/lenet.onnx -D models/input.json --target resources
run_args = ezkl.PyRunArgs()
run_args.scale = 5
run_args.bits = 19
run_args.logrows = 20
