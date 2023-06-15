import ezkl

# Output of:
# ezkl calibrate-settings -M models/lenet.onnx -D models/input.json --target resources
# is:
# Scale: 5
# Bits: 19
# LogRows: 20
#
# But if we evaluate on the entire test set, it turns out that we need more bits when evaluated on the entire test set.
run_args = ezkl.PyRunArgs()
run_args.scale = 5
run_args.bits = 21
run_args.logrows = 22
