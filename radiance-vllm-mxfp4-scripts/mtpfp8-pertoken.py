import json, sys
cfg = json.load(open(sys.argv[1]))
n = 0
for name, spec in cfg["quantization_config"]["layer_quant_config"].items():
    it = spec.get("input_tensors")
    if not name.startswith("mtp") or not it:
        continue
    it["qscheme"]      = "per_channel"
    it["ch_axis"]      = 0
    it["observer_cls"] = "PerChannelMinMaxObserver"
    n += 1
    json.dump(cfg, open(sys.argv[2], "w"), indent=2)
print(f"rewrote {n} mtp input_tensors to per_channel")  # expect 8

