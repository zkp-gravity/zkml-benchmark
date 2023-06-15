import ezkl

# Output of:
# ezkl calibrate-settings -M models/lenet.onnx -D models/input.json --target resources --settings-path settings.json
# But fails with:
# error: constraint not satisfied

#   Cell layout in region 'model':
#     | Offset | A1 | A2 |
#     +--------+----+----+
#     | 295518 | x0 | x1 | <--{ Gate 'RANGE' applied here

#   Constraint '':
#     ((S6 * (0x2 - S6)) * (0x3 - S6)) * (x0 - x1) = 0

#   Assigned cell values:
#     x0 = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593effe0744
#     x1 = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593effdffe6
run_args = ezkl.PyRunArgs()
# run_args.scale = 5
# run_args.bits = 19
# run_args.logrows = 20

# Works on first test example
run_args.scale = 7
run_args.bits = 21
run_args.logrows = 22
