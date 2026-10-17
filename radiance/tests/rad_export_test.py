#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""tools/rad-export without a GPU, a docker daemon or a network.

The checks that need only the standard library always run: PEFT's pattern and scale rules, every
refusal an adapter can meet before a tensor is read (from safetensors headers this test writes
itself), Hub ids from untrusted files, the docker command line, output naming, and recipe
selection over data/recipes. The merge arithmetic needs torch and safetensors and runs when they
import; against PEFT's own merge_and_unload when peft and transformers import too.

    python3 tests/rad_export_test.py
"""
import importlib.machinery, importlib.util, json, math, os, struct, sys, tempfile, types

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
loader = importlib.machinery.SourceFileLoader("rad_export", os.path.join(ROOT, "tools", "rad-export"))
rx = importlib.util.module_from_spec(importlib.util.spec_from_loader("rad_export", loader))
loader.exec_module(rx)

failures = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def refused(fn, word=None):
    """Whether fn() stops with an error (and, with `word`, whether the message names it)."""
    import io, contextlib
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            fn()
    except SystemExit as e:
        return e.code not in (0, None) and (word is None or word in err.getvalue())
    return False


SIZE = {"BF16": 2, "F16": 2, "F32": 4}


def write_st(path, tensors):
    """A safetensors file of name -> (dtype, shape), zero-filled: enough for header checks."""
    hdr, off = {}, 0
    for k, (dt, shape) in tensors.items():
        n = SIZE[dt] * math.prod(shape)
        hdr[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + bytes(off))


def st_base(d, prefix="model.", index=True):
    os.makedirs(d)
    write_st(f"{d}/a.safetensors", {f"{prefix}layers.0.self_attn.q_proj.weight": ("BF16", [64, 32]),
                                    f"{prefix}layers.0.mlp.down_proj.weight": ("BF16", [32, 48])})
    write_st(f"{d}/b.safetensors", {f"{prefix}norm.weight": ("BF16", [32])})
    if index:
        json.dump({"weight_map": {f"{prefix}layers.0.self_attn.q_proj.weight": "a.safetensors",
                                  f"{prefix}layers.0.mlp.down_proj.weight": "a.safetensors",
                                  f"{prefix}norm.weight": "b.safetensors"}},
                  open(f"{d}/model.safetensors.index.json", "w"))
    json.dump({"_name_or_path": "org/Tiny", "vocab_size": 100}, open(f"{d}/config.json", "w"))


def st_adapter(d, cfg, tensors=None, q_rank=4, d_rank=4):
    os.makedirs(d)
    pre = "base_model.model.model.layers.0."
    t = {pre + "self_attn.q_proj.lora_A.weight": ("F32", [q_rank, 32]),
         pre + "self_attn.q_proj.lora_B.weight": ("F32", [64, q_rank]),
         pre + "mlp.down_proj.lora_A.weight": ("F32", [d_rank, 48]),
         pre + "mlp.down_proj.lora_B.weight": ("F32", [32, d_rank])}
    t.update(tensors or {})
    write_st(f"{d}/adapter_model.safetensors", t)
    json.dump({"peft_type": "LORA", "r": 4, "lora_alpha": 8, "base_model_name_or_path": "org/Tiny", **cfg},
              open(f"{d}/adapter_config.json", "w"))


# ================================================================== standard library only

# PEFT's pattern rule: a key matches the end of the module name on a dot boundary
check("pattern: a leaf name matches", rx.pattern_key({"down_proj": 8}, "model.layers.0.mlp.down_proj") == "down_proj")
check("pattern: a parent module does not match its children",
      rx.pattern_key({"self_attn": 8}, "model.layers.0.self_attn.q_proj") is None)
check("pattern: layers.1 does not match layers.11",
      rx.pattern_key({"layers.1.self_attn.q_proj": 8}, "model.layers.11.self_attn.q_proj") is None)
check("pattern: a full path matches itself",
      rx.pattern_key({"model.layers.1.self_attn.q_proj": 8}, "model.layers.1.self_attn.q_proj") is not None)
cfg = {"r": 4, "lora_alpha": 8, "rank_pattern": {"down_proj": 8}, "alpha_pattern": {"down_proj": 4}}
check("scale: alpha / r", rx.scale_of(cfg, "model.layers.0.self_attn.q_proj") == (4, 2.0))
check("scale: rank_pattern and alpha_pattern", rx.scale_of(cfg, "model.layers.0.mlp.down_proj") == (8, 0.5))
check("scale: rsLoRA alpha / math.sqrt(r)",
      rx.scale_of({"r": 7, "lora_alpha": 3, "use_rslora": True}, "x.q_proj") == (7, 3 / math.sqrt(7)))

with tempfile.TemporaryDirectory() as tmp:
    st_base(f"{tmp}/base")
    st_adapter(f"{tmp}/ad", {"rank_pattern": {"down_proj": 8}}, d_rank=8)
    pairs, fifo = rx.plan_merge(f"{tmp}/ad", f"{tmp}/base")
    check("plan: each pair names its base weight and scale",
          sorted(pairs) == ["model.layers.0.mlp.down_proj.weight", "model.layers.0.self_attn.q_proj.weight"] and
          pairs["model.layers.0.mlp.down_proj.weight"][2] == 1.0 and not fifo, pairs)
    st_base(f"{tmp}/vbase", prefix="model.language_model.")
    pairs, _ = rx.plan_merge(f"{tmp}/ad", f"{tmp}/vbase")
    check("plan: a VLM keeps the text model under model.language_model.",
          "model.language_model.layers.0.mlp.down_proj.weight" in pairs)

    def adapter(name, cfg=None, **kw):
        st_adapter(f"{tmp}/{name}", cfg or {}, **kw)
        return lambda: rx.plan_merge(f"{tmp}/{name}", f"{tmp}/base")

    check("refuse: embedding LoRA", refused(adapter("emb", tensors={
        "base_model.model.model.embed_tokens.lora_embedding_A": ("F32", [4, 10])}), "embedding"))
    check("refuse: a trained bias", refused(adapter("bias", tensors={
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.bias": ("F32", [64])})))
    check("refuse: DoRA", refused(adapter("dora", {"use_dora": True}), "DoRA"))
    check("refuse: modules_to_save", refused(adapter("mts", {"modules_to_save": ["lm_head"]}), "modules_to_save"))
    check("refuse: lora_bias", refused(adapter("lb", {"lora_bias": True}), "bias"))
    check("refuse: no r", refused(adapter("nor", {"r": None}), "r is None"))
    check("refuse: a rank the config does not give the module", refused(adapter("rk", d_rank=8), "rank 4"))
    check("refuse: a target the base lacks", refused(adapter("kp", tensors={
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": ("F32", [4, 32]),
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight": ("F32", [64, 4])}), "k_proj"))
    check("refuse: lora_A without lora_B", refused(adapter("nob", tensors={
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": ("F32", [4, 32])}), "without its lora_B"))
    check("refuse: a delta that does not fit the weight", refused(adapter("shape", tensors={
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": ("F32", [48, 4])}), "the weight"))
    check("refuse: a convolution's LoRA", refused(adapter("conv", tensors={
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": ("F32", [4, 32, 1, 1])}), "rank-4"))
    os.makedirs(f"{tmp}/bad")
    open(f"{tmp}/bad/adapter_config.json", "w").write("{not json")
    check("refuse: an adapter config that is not JSON", refused(lambda: rx.plan_merge(f"{tmp}/bad", f"{tmp}/base")))

    # the base: the index decides which shard holds a tensor; without one, a duplicate is refused
    st_base(f"{tmp}/dup", index=False)
    write_st(f"{tmp}/dup/c.safetensors", {"model.norm.weight": ("BF16", [32])})
    check("refuse: a tensor in two shards with no index",
          refused(lambda: rx.weight_files(f"{tmp}/dup"), "model.norm.weight"))
    write_st(f"{tmp}/base/stray.safetensors", {"model.norm.weight": ("BF16", [32])})
    check("base: a shard the index does not name is not read", sorted(rx.weight_files(f"{tmp}/base")) ==
          ["a.safetensors", "b.safetensors"])
    check("refuse: a base folder that is not a checkpoint",
          refused(lambda: rx.checkpoint_dir(f"{tmp}/ad", "--base", False), "not a checkpoint"))

    # Hub ids come from files this tool did not write
    check("hub id: ORG/NAME", rx.hub_id("org/Tiny-2B.v1") == "org/Tiny-2B.v1")
    check("hub id: paths and globs are not ids",
          not any(rx.hub_id(s) for s in ("../x", "org/../x", "org/a/b", "org", "", "/abs/x", "org/*", "org/..")))
    snap = f"{tmp}/hub/models--org--Tiny/snapshots/abc"
    os.makedirs(snap)
    json.dump({}, open(f"{snap}/config.json", "w"))
    json.dump({"weight_map": {"x": "model-00001-of-00001.safetensors"}}, open(f"{snap}/model.safetensors.index.json", "w"))
    check("hub cache: partial snapshot skipped", rx.hub_snapshot("org/Tiny", [f"{tmp}/hub"]) is None)
    open(f"{snap}/model-00001-of-00001.safetensors", "w").write("x")
    check("hub cache: complete snapshot found", rx.hub_snapshot("org/Tiny", [f"{tmp}/hub"]) == snap)
    check("hub cache: an id that climbs out of the cache is not looked up",
          rx.hub_snapshot("org/../../hub/models--org--Tiny", [f"{tmp}/hub"]) is None)

    # a drafter the recipe names only as a placeholder is not guessed from the base model's org
    a0 = types.SimpleNamespace(draft=None)
    check("drafter: a placeholder asks for --draft",
          refused(lambda: rx.resolve_draft(a0, {"hub_cache": [f"{tmp}/hub"]},
                                           {"name": "x", "draft": "MiniCPM5-2B-DSpark checkpoint"}, False), "--draft"))
    check("drafter: a named repository comes from the Hub cache",
          rx.resolve_draft(a0, {"hub_cache": [f"{tmp}/hub"]}, {"name": "x", "draft": "org/Tiny"}, False) == snap)

    # the docker command: the caller's user, no network, no pull, only the named paths
    be = rx.Backend(types.SimpleNamespace(rad_convert=None, image="img:1", memory="8g", recipes_dir=None), {})
    be.native = None
    be.docker = lambda: "docker"
    cmd = be.command(["SRC", "-o", "OUT"], {"/w/out": False, "/w/src": True,
                                            "/h/models--o--d/snapshots/s1": True})
    j = " ".join(cmd)
    check("docker: no network, no pull, the caller's uid",
          "--network none" in j and "--pull never" in j and f"-u {os.getuid()}:{os.getgid()}" in j and "--memory 8g" in j, j)
    check("docker: inputs read-only, the output folder writable, a snapshot by its repository",
          "type=bind,src=/w/src,dst=/w/src,readonly" in j and "type=bind,src=/w/out,dst=/w/out " in j + " " and
          "src=/h/models--o--d,dst=/h/models--o--d,readonly" in j, j)
    check("docker: the image's rad-convert, then the arguments",
          cmd[-5:] == ["/opt/radiance/bin/rad-convert", "img:1", "SRC", "-o", "OUT"], cmd[-5:])
    check("docker: a path that cannot be mounted is refused", refused(lambda: be.command([], {"/a,b": True}), "/a,b"))

check("name: the folder, without an export suffix", rx.out_stem("/x/MyTune-lora") == "MyTune")
check("name: a trainer's generic folder takes its parent's name",
      rx.out_stem("/x/my-run/checkpoint-400") == "my-run-checkpoint-400")

with tempfile.TemporaryDirectory() as tmp:
    json.dump({"model": {"vocab": {"a": 0, "b": 5}}, "added_tokens": [{"id": 9, "content": "<x>"}]},
              open(f"{tmp}/tokenizer.json", "w"))
    check("tokenizer: max id + 1 over the vocabulary and added tokens", rx.tokenizer_size(f"{tmp}/tokenizer.json") == 10)

# every shipped recipe parses; auto picks the stock model's bf16 recipe or asks
recipes = [rx.read_recipe(os.path.join(ROOT, "data", "recipes", f)) for f in sorted(os.listdir(os.path.join(ROOT, "data", "recipes"))) if f.endswith(".recipe")]
check(f"recipes: {len(recipes)} parse", len(recipes) > 0 and all(r["name"] for r in recipes))
by = {r["name"]: r for r in recipes}
if "minicpm5-2b-bf16-dspark" in by:
    r = by["minicpm5-2b-bf16-dspark"]
    check("recipe: minicpm5 dspark names its drafter", bool(r["draft"]), r)
    pick, _ = rx.pick_recipe(types.SimpleNamespace(recipe="auto"), {}, recipes, ["openbmb/MiniCPM5-2B", "MiniCPM5-2B"])
    check("auto: MiniCPM5-2B -> minicpm5-2b-bf16-dspark", pick["name"] == "minicpm5-2b-bf16-dspark", pick["name"])
if "q38-27b-fp8block-df2" in by:
    check("recipe: a usage line's --set keys are read",
          "quantization_config.weight_block_size=128 128" in by["q38-27b-fp8block-df2"]["set"])
check("auto: an unknown model asks for --recipe",
      refused(lambda: rx.pick_recipe(types.SimpleNamespace(recipe="auto"), {}, recipes, ["org/NoSuchModel-9B", "NoSuchModel-9B"])))
pin, _ = rx.pick_recipe(types.SimpleNamespace(recipe="auto"), {"recipes": {"NoSuchModel": recipes[0]["name"]}}, recipes, ["org/NoSuchModel-9B"])
check("auto: the config pins a recipe", pin["name"] == recipes[0]["name"])
# Qwen3.8-Flash-Next's recipes include calibrated ones: auto picks another of them, or asks
fn = ["Qwen/Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next"]
got = []
check("auto: a model with calibrated recipes", any(r["calib"] and fn[1].lower() in r["summary"].lower() for r in recipes))
refused(lambda: got.append(rx.pick_recipe(types.SimpleNamespace(recipe="auto"), {}, recipes, fn)[0]))
check("auto: a recipe that needs calibration is never picked", not got or not got[0]["calib"], got)

# ================================================================== the merge arithmetic (torch)

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None
    print("skip merge arithmetic: torch and safetensors do not import here")

if torch is not None:
    def make_base(d, prefix="model.", fifo=False):
        torch.manual_seed(0)
        w = {f"{prefix}layers.0.self_attn.q_proj.weight": torch.randn(64, 32, dtype=torch.bfloat16),
             f"{prefix}layers.0.mlp.down_proj.weight": torch.randn(32, 48, dtype=torch.bfloat16),
             f"{prefix}norm.weight": torch.ones(32, dtype=torch.bfloat16)}
        if fifo:
            w = {k: (v.T.contiguous() if v.dim() == 2 else v) for k, v in w.items()}
        os.makedirs(d)
        save_file(dict(list(w.items())[:2]), os.path.join(d, "a.safetensors"))
        save_file(dict(list(w.items())[2:]), os.path.join(d, "b.safetensors"))
        json.dump({"_name_or_path": "org/Tiny", "vocab_size": 100}, open(os.path.join(d, "config.json"), "w"))
        open(os.path.join(d, "tokenizer.json"), "w").write('{"model": {"vocab": {"a": 0}}}')
        open(os.path.join(d, "pytorch_model.bin"), "w").write("not copied")
        return w

    def make_adapter(d, cfg, ranks=(4, 4), dtype=torch.float32):
        torch.manual_seed(1)
        q, dn = ranks
        p = "base_model.model.model.layers.0."
        t = {p + "self_attn.q_proj.lora_A.weight": torch.randn(q, 32, dtype=dtype),
             p + "self_attn.q_proj.lora_B.weight": torch.randn(64, q, dtype=dtype),
             p + "mlp.down_proj.lora_A.weight": torch.randn(dn, 48, dtype=dtype),
             p + "mlp.down_proj.lora_B.weight": torch.randn(32, dn, dtype=dtype)}
        os.makedirs(d)
        save_file(t, os.path.join(d, "adapter_model.safetensors"))
        json.dump({"peft_type": "LORA", "r": 4, "lora_alpha": 8, "base_model_name_or_path": "org/Tiny", **cfg},
                  open(os.path.join(d, "adapter_config.json"), "w"))
        return t

    def merged(out):
        m = {}
        idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
        for f in set(idx.values()):
            m.update(load_file(os.path.join(out, f)))
        return m, idx

    def peft_like(base, ad, scales, prefix="model.", fifo=False):
        """PEFT's CPU merge: delta (B @ A) * scale in fp32, rounded to a half-precision adapter's
        dtype, added to the weight and rounded once to its dtype."""
        e = {}
        for k, w in base.items():
            mod = k[len(prefix):-len(".weight")]
            a = ad.get(f"base_model.model.model.{mod}.lora_A.weight")
            if a is None:
                e[k] = w
                continue
            b = ad[f"base_model.model.model.{mod}.lora_B.weight"]
            d = (b.float() @ a.float()) * scales[mod]
            if fifo:
                d = d.T
            if b.dtype != torch.float32:
                d = d.to(b.dtype)
            e[k] = (w.float() + d.float()).to(w.dtype)
        return e

    def same(got, want):
        return sorted(got) == sorted(want) and all(torch.equal(got[k], want[k]) for k in want)

    with tempfile.TemporaryDirectory() as tmp:
        base = make_base(f"{tmp}/base")
        ad = make_adapter(f"{tmp}/ad", {"rank_pattern": {"down_proj": 8}, "alpha_pattern": {"down_proj": 4}}, ranks=(4, 8))
        rx.merge_lora(f"{tmp}/ad", f"{tmp}/base", f"{tmp}/out")
        got, idx = merged(f"{tmp}/out")
        check("merge: alpha/r, rank_pattern and alpha_pattern",
              same(got, peft_like(base, ad, {"layers.0.self_attn.q_proj": 2.0, "layers.0.mlp.down_proj": 0.5})))
        check("merge: the index names every tensor once", sorted(idx) == sorted(base))
        check("merge: config and tokenizer copied, other weight files not",
              os.path.exists(f"{tmp}/out/tokenizer.json") and os.path.exists(f"{tmp}/out/config.json") and
              not os.path.exists(f"{tmp}/out/pytorch_model.bin"))

        ad2 = make_adapter(f"{tmp}/ad2", {"use_rslora": True})
        rx.merge_lora(f"{tmp}/ad2", f"{tmp}/base", f"{tmp}/out2")
        s = 8 / math.sqrt(4)
        check("merge: rsLoRA alpha/sqrt(r)",
              same(merged(f"{tmp}/out2")[0], peft_like(base, ad2, {"layers.0.self_attn.q_proj": s, "layers.0.mlp.down_proj": s})))

        ad3 = make_adapter(f"{tmp}/ad3", {}, dtype=torch.bfloat16)
        rx.merge_lora(f"{tmp}/ad3", f"{tmp}/base", f"{tmp}/out3")
        got3 = merged(f"{tmp}/out3")[0]
        want3 = peft_like(base, ad3, {"layers.0.self_attn.q_proj": 2.0, "layers.0.mlp.down_proj": 2.0})
        check("merge: a bf16 adapter's delta rounded to bf16 first, as PEFT does", same(got3, want3))

        vbase = make_base(f"{tmp}/vbase", prefix="model.language_model.")
        rx.merge_lora(f"{tmp}/ad2", f"{tmp}/vbase", f"{tmp}/vout")
        check("merge: model.language_model. prefix",
              same(merged(f"{tmp}/vout")[0], peft_like(vbase, ad2, {"layers.0.self_attn.q_proj": s, "layers.0.mlp.down_proj": s},
                                                      prefix="model.language_model.")))

        fbase = make_base(f"{tmp}/fbase", fifo=True)
        ad4 = make_adapter(f"{tmp}/ad4", {"fan_in_fan_out": True})
        rx.merge_lora(f"{tmp}/ad4", f"{tmp}/fbase", f"{tmp}/fout")
        check("merge: fan_in_fan_out (Conv1D) weights take the transposed delta",
              same(merged(f"{tmp}/fout")[0], peft_like(fbase, ad4, {"layers.0.self_attn.q_proj": 2.0, "layers.0.mlp.down_proj": 2.0},
                                                      fifo=True)))

        json.dump({"model": {"vocab": {"a": 0, "b": 1}}, "added_tokens": [{"id": 7}]}, open(f"{tmp}/ad2/tokenizer.json", "w"))
        open(f"{tmp}/ad2/chat_template.jinja", "w").write("tuned")
        rx.merge_lora(f"{tmp}/ad2", f"{tmp}/base", f"{tmp}/out5")
        check("merge: the adapter's tokenizer and chat template replace the base's",
              json.load(open(f"{tmp}/out5/tokenizer.json"))["added_tokens"][0]["id"] == 7 and
              open(f"{tmp}/out5/chat_template.jinja").read() == "tuned")
        json.dump({"model": {"vocab": {"a": 0}}, "added_tokens": [{"id": 100}]}, open(f"{tmp}/ad2/tokenizer.json", "w"))
        check("refuse: an adapter tokenizer larger than the base's embedding",
              refused(lambda: rx.merge_lora(f"{tmp}/ad2", f"{tmp}/base", f"{tmp}/out6"), "101 tokens"))

    try:
        import peft, transformers
    except ImportError:
        peft = None
        print("skip PEFT parity: peft and transformers do not import here")
    if peft is not None:
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(3)
            tc = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
            transformers.LlamaForCausalLM(tc).to(torch.bfloat16).save_pretrained(f"{tmp}/base", safe_serialization=True)
            # fp32 adapter weights (PEFT's default for a bf16 base) and bf16 ones
            for i, autocast in enumerate((True, False)):
                m = transformers.LlamaForCausalLM.from_pretrained(f"{tmp}/base", torch_dtype=torch.bfloat16)
                lc = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"],
                                     rank_pattern={"down_proj": 8}, alpha_pattern={"v_proj": 3})
                pm = peft.get_peft_model(m, lc, autocast_adapter_dtype=autocast)
                with torch.no_grad():
                    for n, prm in pm.named_parameters():
                        if "lora_" in n:
                            prm.copy_(torch.randn_like(prm) * 0.1)
                pm.save_pretrained(f"{tmp}/ad{i}")
                ref = pm.merge_and_unload().state_dict()
                rx.merge_lora(f"{tmp}/ad{i}", f"{tmp}/base", f"{tmp}/out{i}")
                got, _ = merged(f"{tmp}/out{i}")
                keys = [k for k in ref if k in got]
                dt = "fp32" if autocast else "bf16"
                check(f"PEFT parity ({dt} adapter): merge_and_unload's weights, bit for bit",
                      len(keys) == len(got) and all(torch.equal(got[k], ref[k]) for k in keys),
                      [k for k in keys if not torch.equal(got[k], ref[k])][:3])

print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all passed'}")
sys.exit(1 if failures else 0)
