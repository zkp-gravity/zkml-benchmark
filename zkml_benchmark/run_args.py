import ezkl

# In theory, we can do:
# ezkl gen-circuit-params -M models/lenet.onnx --calibration-data models/input.json --calibration-target resources --circuit-params-path circuit.json
# (using ezkl 0.2.0) 
# This gives: "scale":5, "bits":19, "logrows":20
# However, when using this, measure_accuracy.py prints:
# At the selected lookup bits and fixed point scale, the largest input to a lookup table is too large to be represented (max: 270886, bits: 19, scale: 5).
# Increase the lookup bits to [20]. The current scale cannot be decreased enough to fit the largest lookup input.
# So, I decreased the scale to 4.
run_args = ezkl.PyRunArgs()
run_args.scale = 4
run_args.bits = 19
run_args.logrows = 20