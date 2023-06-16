import ezkl

# Output of:
# ezkl calibrate-settings -M models/lenet.onnx -D models/input.json --target resources
# is:
# Scale: 5
# Bits: 16
# LogRows: 17
#
# But if we evaluate on the entire test set, it turns out that we need more bits when evaluated on the entire test set.
# Also, we can be even faster if we use a smaller scale.
run_args = ezkl.PyRunArgs()
run_args.scale = 4
run_args.bits = 15
run_args.logrows = 16
