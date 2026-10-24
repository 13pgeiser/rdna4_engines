// Live size display for the Radiance MiniMax node: mirrors the Python tier math
// (radiance_node.py `_tier_size` / `_snap_text`) so the node TITLE shows the
// exact WxH the server will receive -- "480p 16:9 -> 864x480". Display only:
// Python recomputes at execute() and is the authority.
import { app } from "../../scripts/app.js";

const TIERS = { "144p": 144, "240p": 240, "360p": 360, "480p": 480, "540p": 540, "576p": 576, "720p": 720 };
const SIZE_MULT = 32, MIN_SIDE = 256, MAX_AREA = 1344 * 768, MAX_ASPECT = 4.0;

const snapUp = (v) => Math.max(MIN_SIDE, Math.floor((v + SIZE_MULT / 2) / SIZE_MULT) * SIZE_MULT);

function fit(w, h, cap) {
  const s = Math.sqrt(cap / (w * h));
  let fw = Math.max(MIN_SIDE, Math.floor((w * s) / SIZE_MULT) * SIZE_MULT);
  let fh = Math.max(MIN_SIDE, Math.floor((h * s) / SIZE_MULT) * SIZE_MULT);
  while (fw * fh > cap) {
    if (fw >= fh && fw > MIN_SIDE) fw -= SIZE_MULT;
    else if (fh > MIN_SIDE) fh -= SIZE_MULT;
    else break;
  }
  if (Math.max(fw, fh) > MAX_ASPECT * Math.min(fw, fh)) {
    if (fw >= fh) fw = Math.floor((MAX_ASPECT * fh) / SIZE_MULT) * SIZE_MULT;
    else fh = Math.floor((MAX_ASPECT * fw) / SIZE_MULT) * SIZE_MULT;
  }
  return [fw, fh];
}

function resolveSize(resolution, aspect, customText) {
  if (resolution === "custom") {
    const m = /^\s*(\d+)\s*[xX×*]\s*(\d+)/.exec(customText || "");
    if (!m) return null;
    let [lw, lh] = [snapUp(+m[1]), snapUp(+m[2])];
    if (lw * lh > MAX_AREA || Math.max(lw, lh) > MAX_ASPECT * Math.min(lw, lh)) [lw, lh] = fit(lw, lh, MAX_AREA);
    return `${lw}x${lh}`;
  }
  const tier = TIERS[resolution];
  if (tier === undefined) return null;
  const [a, b] = (aspect || "16:9").split(":").map(Number);
  const [hi, lo] = [Math.max(a, b), Math.min(a, b)];
  let short = snapUp(tier);
  let long = Math.max(short, snapUp(Math.floor((short * hi) / lo)));
  if (long * short > MAX_AREA) [long, short] = fit(long, short, MAX_AREA);
  return a >= b ? `${long}x${short}` : `${short}x${long}`;
}

// The loras DROPDOWN. Its options come from THE SERVER (GET <server_url>/v1/loras
// -- radiance sends CORS headers unless --no-cors) because the adapters live there
// and differ per box; the schema ships only "none". Fetched on node creation, on
// workflow load, when server_url is edited, and by the "refresh lora list" button.
// The fetch replaces the OPTION LIST and never touches the selected value (older
// builds joined every name into the text box -- one missed click and the job ran
// with all adapters at once). The only value write: a selection the (re)fetched
// server no longer offers resets to "none"; free text an older graph saved
// ('film:0.8, ink') survives and still executes -- Python parses it verbatim.
function loraDropdown(node) {
  if (!node.widgets) return;
  const widget = (name) => node.widgets.find((w) => w.name === name);
  const dropNames = ["loras", "lora2"];
  const fetchLoras = (announce) => {
    const base = (widget("server_url")?.value || "").trim().replace(/\/+$/, "");
    if (!dropNames.some((n) => widget(n)) || !base) return;
    fetch(`${base}/v1/loras`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((j) => {
        const names = (j.data || []).map((d) => d.id);
        for (const w of [widget("loras"), widget("lora2")]) {
          if (!w) continue;
          // the combo's option list, however this frontend version stores it
          if (!w.options) w.options = {};
          w.options.values = ["none", ...names];
          if (Array.isArray(w.options)) w.options.length = 0, w.options.push("none", ...names);
          // a plain-name selection the refetched server no longer offers is gone
          // -- back to "none". Values holding ',' or ':' are legacy free text
          // from pre-dropdown graphs (or a colon-carrying name): never rewritten.
          const cur = String(w.value ?? "");
          if (!["none", ...names].includes(cur) && !cur.includes(",") && !cur.includes(":")) {
            w.value = "none";
          }
        }
        node.setDirtyCanvas?.(true, true);
        if (announce) {
          alert(names.length
            ? `loras dropdown now lists ${names.length} adapter(s):\n\n${names.join("\n")}`
            : "this radiance server lists no LoRAs\n(start it with --lora-dir)");
        }
      })
      .catch((e) => {
        console.warn(`[RadianceMiniMax] cannot list LoRAs from ${base}:`, e);
        if (announce) alert(`cannot list LoRAs from ${base}: ${e}\n(the server must be reachable from this browser; --no-cors breaks this)`);
      });
  };
  node._radianceFetchLoras = () => fetchLoras(false);
  if (!node.widgets.some((w) => w.name === "refresh lora list")) {
    const w = node.addWidget("button", "refresh lora list", null, () => fetchLoras(true));
    w.serialize = false;  // display helper, never part of the prompt
  }
  // a graph saved against the old free-text widget may hold "" -- "none" is the
  // dropdown's off state; real text is left alone and still executes.
  let fixed = false;
  for (const n of dropNames) {
    const w = widget(n);
    if (w && !String(w.value ?? "").trim()) {
      w.value = "none";
      fixed = true;
    }
  }
  if (fixed) node.setDirtyCanvas?.(true, false);
  fetchLoras(false);
}

function refresh(node) {
  const widget = (name) => node.widgets?.find((w) => w.name === name);
  const res = widget("resolution")?.value;
  const aspect = widget("aspect")?.value ?? "16:9";
  if (!res) return;
  const size = resolveSize(res, aspect, widget("custom_size")?.value ?? "");
  node.title = size ? (res === "custom" ? `custom -> ${size}` : `${res} ${aspect} -> ${size}`)
                    : `${res} (not a size)`;
  node.setDirtyCanvas?.(true, false);
}

app.registerExtension({
  name: "Comfy.RadianceMiniMax.SizeDisplay",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "RadianceMiniMax") return;
    const wrap = (node, name, extra) => {
      const w = node.widgets?.find((x) => x.name === name);
      if (!w || w._radianceWrapped) return;
      w._radianceWrapped = true;
      const inner = w.callback;
      w.callback = function (...args) {
        const r = inner?.apply(this, args);
        refresh(node);
        extra?.(node);
        return r;
      };
    };
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      for (const name of ["resolution", "aspect", "custom_size"]) wrap(this, name);
      // pointing the node at another box swaps the LoRA list too (silent fetch)
      wrap(this, "server_url", (n) => n._radianceFetchLoras?.());
      loraDropdown(this);
      // DynamicCombo.Input has no default knob; fresh nodes would land on the first
      // option (144p). Nudge them to 480p -- on a workflow LOAD onConfigure re-applies
      // the stored value right after this, so saved graphs keep what they had.
      const resW = this.widgets?.find((w) => w.name === "resolution");
      if (resW && resW.value !== "480p") {
        resW.value = "480p";
        try { resW.callback?.call(resW, "480p", undefined, this); } catch (e) { /* display nudge only */ }
      }
      if (!this._radianceWidgetHook) {  // DynamicCombo adds custom_size on toggle
        this._radianceWidgetHook = true;
        const added = this.onWidgetAdded;
        this.onWidgetAdded = function (w) {
          const r = added?.apply(this, arguments);
          if (w && ["resolution", "aspect", "custom_size"].includes(w.name)) wrap(this, w.name);
          refresh(this);
          return r;
        };
      }
      refresh(this);
      return r;
    };
    const onConfigured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (d) {
      const r = onConfigured?.apply(this, arguments);
      for (const name of ["resolution", "aspect", "custom_size"]) wrap(this, name);
      wrap(this, "server_url", (n) => n._radianceFetchLoras?.());
      loraDropdown(this);  // serialize=false button => dropped on save; re-add on load
      refresh(this);
      return r;
    };
  },
});
