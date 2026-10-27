// The Radiance MiniMax node's frontend: the size a request will carry in the node's title
// ("480p 16:9 -> 864x480"), computed as radiance_node.py's _tier_size and _custom_size compute it
// at execute(), and the adapter lists, fetched from the server.
import { app } from "../../scripts/app.js";

const TIERS = { "144p": 144, "240p": 240, "360p": 360, "480p": 480, "540p": 540, "576p": 576,
                "720p": 720 };
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
    if (lw * lh > MAX_AREA || Math.max(lw, lh) > MAX_ASPECT * Math.min(lw, lh)) {
      [lw, lh] = fit(lw, lh, MAX_AREA);
    }
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

// The adapter lists are the server's (GET <server_url>/v1/loras, which needs the server's CORS
// headers), so the schema ships only "none". They are fetched when the node is created or
// loaded, when server_url changes, and by the "refresh lora list" button. A fetch replaces the
// options; a selection the server no longer offers goes back to "none".
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
        const values = ["none", ...names];
        for (const w of dropNames.map(widget)) {
          if (!w) continue;
          // A combo keeps its options as an array or as { values }, by frontend version.
          if (Array.isArray(w.options)) w.options.splice(0, w.options.length, ...values);
          else (w.options ??= {}).values = values;
          if (!values.includes(w.value)) w.value = "none";
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
        if (announce) {
          alert(`cannot list LoRAs from ${base}: ${e}\n` +
                "(the server must be reachable from this browser, and not run with --no-cors)");
        }
      });
  };
  node._radianceFetchLoras = () => fetchLoras(false);
  if (!node.widgets.some((w) => w.name === "refresh lora list")) {
    const w = node.addWidget("button", "refresh lora list", null, () => fetchLoras(true));
    w.serialize = false;  // not an input of the node
  }
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
      wrap(this, "server_url", (n) => n._radianceFetchLoras?.());
      loraDropdown(this);
      // DynamicCombo.Input takes no default, so a new node would start on 144p. A loaded node
      // gets its saved value back from onConfigure, which runs after this.
      const resW = this.widgets?.find((w) => w.name === "resolution");
      if (resW && resW.value !== "480p") {
        resW.value = "480p";
        try { resW.callback?.call(resW, "480p", undefined, this); } catch (e) { /* the title only */ }
      }
      if (!this._radianceWidgetHook) {  // the DynamicCombo adds custom_size when it is chosen
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
      loraDropdown(this);  // the button is not saved with the workflow, so it is added again
      refresh(this);
      return r;
    };
  },
});
