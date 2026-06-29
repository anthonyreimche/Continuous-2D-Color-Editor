// Continuous 2D Color Editor for Safelight
// Copyright (c) 2026 Anthony Reimche. MIT License.
//
// A Capture One-style continuous colour editor. Instead of eight fixed hue
// buckets, you drop points anywhere on a hue/saturation wheel; each point
// selects a smooth 2D region of colour (a disc on the chromaticity plane) and
// reshapes its hue, saturation and luminance. The per-pixel work runs on the GPU
// as a registered processing stage; the points are stored per photo in the
// develop param bag, so they save with the edit and undo/redo like any slider.
//
// Colour science is deliberately identical to Safelight's built-in HSL mixer:
// near-neutral pixels are protected, hue rotates in HSL, saturation is luminance-
// anchored, and luminance is a hue-preserving gain.

const STAGE_ID = "color2d-editor.color";
const PANEL_ID = "color2d-editor.panel";
const MAX_POINTS = 8;
const TWO_PI = Math.PI * 2;

// Param-bag key helpers. Each point is two vec4 uniforms:
//   A = (targetHueTurns, targetSat, rangeNorm, softnessNorm)
//   B = (hueShift, satAdj, lumAdj, active)
const KA = (i) => `${STAGE_ID}.uCP${i}A`;
const KB = (i) => `${STAGE_ID}.uCP${i}B`;
const KCOUNT = `${STAGE_ID}.uCPCount`;
const ZERO4 = [0, 0, 0, 0];

// ---------------------------------------------------------------------------
// GPU processing stage
// ---------------------------------------------------------------------------

// One vec4 pair per point + an active-count for the early-out. Built
// programmatically so the slot count stays in one place.
function buildUniforms() {
  const u = [];
  for (let i = 0; i < MAX_POINTS; i++) {
    u.push({ key: `uCP${i}A`, glslType: "vec4", default: [0, 0, 0, 0] });
    u.push({ key: `uCP${i}B`, glslType: "vec4", default: [0, 0, 0, 0] });
  }
  u.push({ key: "uCPCount", glslType: "int", default: 0 });
  return u;
}

// Format a JS number as a GLSL float literal (always with a decimal point).
function glslFloat(n) {
  let s = String(Math.round(n * 1e4) / 1e4);
  if (!s.includes(".") && !s.includes("e")) s += ".0";
  return s;
}

// Self-contained helpers (the compiler namespaces these names + the uniform
// keys, so nothing here collides with the core shader). The math mirrors the
// core applyHSL so the two tools feel the same. `neutral` (the neutral-protection
// threshold) and `lumGain` (luminance strength) are baked in from preferences —
// changing a pref re-registers the stage, which recompiles with the new values.
function buildHelpers(neutral, lumGain) {
  // smoothstep needs edge0 < edge1, so floor the threshold just above 0.
  const nThresh = glslFloat(Math.max(neutral, 0.002));
  const gain = glslFloat(lumGain);
  return `
vec3 c2d_rgb2hsl(vec3 c) {
  float mx = max(max(c.r, c.g), c.b);
  float mn = min(min(c.r, c.g), c.b);
  float h = 0.0, s = 0.0, l = (mx + mn) * 0.5, d = mx - mn;
  if (d > 1e-5) {
    s = l < 0.5 ? d / (mx + mn) : d / (2.0 - mx - mn);
    if (mx == c.r) h = (c.g - c.b) / d + (c.g < c.b ? 6.0 : 0.0);
    else if (mx == c.g) h = (c.b - c.r) / d + 2.0;
    else h = (c.r - c.g) / d + 4.0;
    h /= 6.0;
  }
  return vec3(h, s, l);
}
float c2d_hue2rgb(float p, float q, float t) {
  if (t < 0.0) t += 1.0;
  if (t > 1.0) t -= 1.0;
  if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
  if (t < 0.5) return q;
  if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
  return p;
}
vec3 c2d_hsl2rgb(vec3 hsl) {
  float h = hsl.x, s = hsl.y, l = hsl.z;
  if (s <= 1e-5) return vec3(l);
  float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
  float p = 2.0 * l - q;
  return vec3(c2d_hue2rgb(p, q, h + 1.0 / 3.0), c2d_hue2rgb(p, q, h), c2d_hue2rgb(p, q, h - 1.0 / 3.0));
}
float c2d_luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// One colour point's contribution. A = (targetHueTurns, targetSat, range, soft),
// B = (hueShift, satAdj, lumAdj, active). Accumulates (hue, sat, lum) offsets
// weighted by a smooth 2D selection on the chromaticity disc. No arrays / dynamic
// indexing — just an explicit call per slot, the most GLSL-portable form.
void c2d_pt(vec4 A, vec4 B, vec2 pPix, inout vec3 acc) {
  if (B.w < 0.5) return; // inactive slot
  float ta = A.x * 6.28318530718;
  vec2 tPt = A.y * vec2(cos(ta), sin(ta));
  float range = max(A.z, 1e-3);
  float soft = clamp(A.w, 0.02, 1.0);
  float w = 1.0 - smoothstep(range * (1.0 - soft), range, length(pPix - tPt));
  acc += vec3(B.x, B.y, B.z) * w;
}

vec3 c2d_apply(vec3 col) {
  if (uCPCount < 1) return col;
  vec3 hsl = c2d_rgb2hsl(clamp(col, 0.0, 1.0));
  float chromaW = smoothstep(0.0, ${nThresh}, hsl.y); // protect near-neutral pixels
  if (chromaW < 1e-4) return col;

  float pa = hsl.x * 6.28318530718;
  vec2 pPix = hsl.y * vec2(cos(pa), sin(pa)); // pixel on the chromaticity disc

  vec3 acc = vec3(0.0); // (hueShift, satAdj, lumAdj)
  c2d_pt(uCP0A, uCP0B, pPix, acc);
  c2d_pt(uCP1A, uCP1B, pPix, acc);
  c2d_pt(uCP2A, uCP2B, pPix, acc);
  c2d_pt(uCP3A, uCP3B, pPix, acc);
  c2d_pt(uCP4A, uCP4B, pPix, acc);
  c2d_pt(uCP5A, uCP5B, pPix, acc);
  c2d_pt(uCP6A, uCP6B, pPix, acc);
  c2d_pt(uCP7A, uCP7B, pPix, acc);

  float hueShift = acc.x * chromaW;
  float satAdj   = acc.y * chromaW;
  float lumAdj   = acc.z * chromaW;

  hsl.x = fract(hsl.x + hueShift * (30.0 / 360.0)); // hue rotates in HSL
  vec3 rgb = c2d_hsl2rgb(hsl);
  if (abs(satAdj) > 1e-4) {                          // perceptual saturation
    float L = c2d_luma(rgb);
    rgb = mix(vec3(L), rgb, clamp(1.0 + satAdj, 0.0, 2.0));
  }
  if (abs(lumAdj) > 1e-4) rgb *= 1.0 + lumAdj * ${gain}; // hue-preserving luminance
  return clamp(rgb, 0.0, 1.0);
}
`;
}

const STAGE_GLSL = `c = c2d_apply(c);`;

// Default tuning (also the Preferences defaults): gain 0.6, threshold 0.10 — the
// same constants the core HSL mixer uses.
const DEFAULT_LUM_STRENGTH = 60; // → gain 0.60
const DEFAULT_NEUTRAL_PROTECT = 10; // → threshold 0.10
// New-point selection defaults (also the Preferences defaults). These map the
// same way the per-point Range/Smooth sliders do: range = v/100 * 1.5, soft = v/100.
const DEFAULT_RANGE = 40; // → disc radius 0.60
const DEFAULT_SMOOTH = 50; // → softness 0.50

function buildStage(lumGain, neutral) {
  return {
    id: STAGE_ID,
    name: "2D Color Editor",
    phase: "display-adjust",
    priority: 10, // colour-shape first, before creative effects (vignette/grain)
    glsl: STAGE_GLSL,
    helpers: buildHelpers(neutral, lumGain),
    uniforms: buildUniforms(),
  };
}

// ---------------------------------------------------------------------------
// Wheel rendering (HSV guide; the radius IS saturation, matching selection)
// ---------------------------------------------------------------------------

function hsvToRgb(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r, g, b;
  switch (((i % 6) + 6) % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function drawWheel(canvas, size) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const img = ctx.createImageData(size, size);
  const d = img.data;
  const r0 = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x - r0) / r0;
      const dy = (y - r0) / r0;
      const rad = Math.hypot(dx, dy);
      const idx = (y * size + x) * 4;
      if (rad > 1) { d[idx + 3] = 0; continue; }
      let ang = Math.atan2(-dy, dx);
      if (ang < 0) ang += TWO_PI;
      const [rr, gg, bb] = hsvToRgb(ang / TWO_PI, Math.min(rad, 1), 1);
      // Slight inner fade so the desaturated centre reads as "no selection".
      d[idx] = rr; d[idx + 1] = gg; d[idx + 2] = bb;
      d[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ---------------------------------------------------------------------------
// Extension entry
// ---------------------------------------------------------------------------

let _settingsUnsub = null; // api.settings.onChange cleanup
let _retimer = null; // debounce timer for stage re-bake

export function activate(api) {
  const React = api.react;
  const h = React.createElement;
  const { Panel, Slider } = api.components;
  const useDevelopStore = api.stores.useDevelopStore;

  // ---- engine tuning (global preferences, baked into the stage GLSL) ----
  const readTuning = () => ({
    lumGain: api.settings.get("lumStrength", DEFAULT_LUM_STRENGTH) / 100,
    neutral: api.settings.get("neutralProtect", DEFAULT_NEUTRAL_PROTECT) / 100,
  });

  let cur = readTuning();
  api.registerProcessingStage(buildStage(cur.lumGain, cur.neutral));

  api.registerSettings({
    title: "2D Color Editor",
    order: 61, // just after the built-in HSL section
    keywords: ["color", "colour", "2d", "luminance", "neutral", "protect"],
    fields: [
      {
        key: "lumStrength",
        label: "Luminance strength",
        hint: "How hard the Luminance slider lifts or drops brightness. 60 matches the built-in HSL mixer.",
        type: "number",
        default: DEFAULT_LUM_STRENGTH,
        min: 0,
        max: 150,
        step: 5,
      },
      {
        key: "neutralProtect",
        label: "Neutral protection",
        hint: "Pixels below this saturation (%) are treated as neutral and left untouched, so greys aren't tinted. 10 matches the built-in HSL mixer.",
        type: "number",
        default: DEFAULT_NEUTRAL_PROTECT,
        min: 0,
        max: 30,
        step: 1,
      },
      {
        key: "defaultRange",
        label: "Default hue/sat range",
        hint: "Selection size a freshly-dropped point starts with — how wide a hue/saturation region it covers. Tune to your editing style.",
        type: "number",
        default: DEFAULT_RANGE,
        min: 5,
        max: 100,
        step: 5,
      },
      {
        key: "defaultSmooth",
        label: "Default smoothness",
        hint: "How soft a freshly-dropped point's selection edge starts. Higher feathers the falloff more.",
        type: "number",
        default: DEFAULT_SMOOTH,
        min: 2,
        max: 100,
        step: 1,
      },
    ],
  });

  // Re-bake the stage when a tuning pref changes. The number fields are sliders
  // that fire per drag-tick, so debounce — a shader recompile per tick would jank.
  _settingsUnsub = api.settings.onChange((key) => {
    if (key !== "lumStrength" && key !== "neutralProtect") return;
    if (_retimer) clearTimeout(_retimer);
    _retimer = setTimeout(() => {
      _retimer = null;
      const next = readTuning();
      if (next.lumGain === cur.lumGain && next.neutral === cur.neutral) return;
      cur = next;
      api.registerProcessingStage(buildStage(cur.lumGain, cur.neutral));
    }, 150);
  });

  // ---- param-bag accessors (per photo, undoable) ----
  const readA = (bag, i) => bag[KA(i)] || ZERO4;
  const readB = (bag, i) => bag[KB(i)] || ZERO4;

  function writeSlots(patch) {
    // Recompute the active count from the freshest state plus this patch, so
    // the shader's early-out stays correct.
    const bag = useDevelopStore.getState().paramBag;
    let count = 0;
    for (let i = 0; i < MAX_POINTS; i++) {
      const b = patch[KB(i)] || readB(bag, i);
      if (b[3] >= 0.5) count++;
    }
    useDevelopStore.getState().setDynParams({ ...patch, [KCOUNT]: count });
  }

  const commit = () => void useDevelopStore.getState().commitEdit("2D Color Editor");

  function clearAll() {
    const patch = {};
    for (let i = 0; i < MAX_POINTS; i++) { patch[KA(i)] = ZERO4; patch[KB(i)] = ZERO4; }
    patch[KCOUNT] = 0;
    useDevelopStore.getState().setDynParams(patch);
    void useDevelopStore.getState().commitEdit("Reset Color Editor");
  }

  // ---- panel ----
  function ColorEditorPanel() {
    const paramBag = useDevelopStore((s) => s.paramBag);
    const [selected, setSelected] = React.useState(-1);
    const [, force] = React.useReducer((n) => n + 1, 0);
    const canvasRef = React.useRef(null);
    const wrapRef = React.useRef(null);
    const dragRef = React.useRef(null);
    const SIZE = 188;

    React.useEffect(() => {
      if (canvasRef.current) drawWheel(canvasRef.current, SIZE);
    }, []);

    const active = [];
    for (let i = 0; i < MAX_POINTS; i++) {
      if (readB(paramBag, i)[3] >= 0.5) active.push(i);
    }
    const sel = active.includes(selected) ? selected : (active.length ? active[0] : -1);

    // Screen position (px, relative to the wheel box) of a point slot.
    const posOf = (i) => {
      const a = readA(paramBag, i);
      const ang = a[0] * TWO_PI;
      const r = Math.min(a[1], 1) * (SIZE / 2);
      return { x: SIZE / 2 + r * Math.cos(ang), y: SIZE / 2 - r * Math.sin(ang) };
    };

    // Pointer -> (hueTurns, sat). getBoundingClientRect (not offsetX/Y) so it
    // survives Windows display scaling and the interface-scale zoom.
    const pointToHS = (clientX, clientY) => {
      const rect = wrapRef.current.getBoundingClientRect();
      const r0 = rect.width / 2;
      const dx = (clientX - rect.left - r0) / r0;
      const dy = (clientY - rect.top - r0) / r0;
      let ang = Math.atan2(-dy, dx);
      if (ang < 0) ang += TWO_PI;
      return { hue: ang / TWO_PI, sat: Math.min(Math.hypot(dx, dy), 1) };
    };

    const onWheelDown = (e) => {
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      // Hit-test existing points first.
      let hit = -1;
      for (const i of active) {
        const p = posOf(i);
        if (Math.hypot(e.clientX - wrapRef.current.getBoundingClientRect().left - p.x,
                       e.clientY - wrapRef.current.getBoundingClientRect().top - p.y) <= 11) { hit = i; break; }
      }
      if (hit === -1) {
        // Add a new point in the first free slot.
        let free = -1;
        for (let i = 0; i < MAX_POINTS; i++) if (readB(paramBag, i)[3] < 0.5) { free = i; break; }
        if (free === -1) return; // all slots used
        const { hue, sat } = pointToHS(e.clientX, e.clientY);
        // New points adopt the user's preferred selection style (Preferences).
        const range = (api.settings.get("defaultRange", DEFAULT_RANGE) / 100) * 1.5;
        const soft = api.settings.get("defaultSmooth", DEFAULT_SMOOTH) / 100;
        writeSlots({ [KA(free)]: [hue, sat, range, soft], [KB(free)]: [0, 0, 0, 1] });
        hit = free;
      }
      setSelected(hit);
      dragRef.current = hit;
    };

    const onWheelMove = (e) => {
      if (dragRef.current == null) return;
      const i = dragRef.current;
      const { hue, sat } = pointToHS(e.clientX, e.clientY);
      const a = readA(paramBag, i);
      writeSlots({ [KA(i)]: [hue, sat, a[2], a[3]] });
    };

    const onWheelUp = () => {
      if (dragRef.current == null) return;
      dragRef.current = null;
      commit();
    };

    // Slider helpers for the selected point.
    const a = sel >= 0 ? readA(paramBag, sel) : ZERO4;
    const b = sel >= 0 ? readB(paramBag, sel) : ZERO4;
    const setA = (idx, val) => { const na = a.slice(); na[idx] = val; writeSlots({ [KA(sel)]: na }); };
    const setB = (idx, val) => { const nb = b.slice(); nb[idx] = val; writeSlots({ [KB(sel)]: nb }); };
    const removeSel = () => { writeSlots({ [KA(sel)]: ZERO4, [KB(sel)]: ZERO4 }); commit(); setSelected(-1); force(); };

    const markers = active.map((i) => {
      const p = posOf(i);
      const isSel = i === sel;
      return h("div", {
        key: i,
        style: {
          position: "absolute", left: p.x + "px", top: p.y + "px",
          width: "13px", height: "13px", marginLeft: "-7px", marginTop: "-7px",
          borderRadius: "50%", boxSizing: "border-box",
          border: isSel ? "2px solid #fff" : "2px solid rgba(255,255,255,0.7)",
          boxShadow: "0 0 0 1.5px rgba(0,0,0,0.6)", background: "transparent",
          pointerEvents: "none",
        },
      });
    });

    const label = (t) => h("div", {
      style: { fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.05em",
        color: "var(--color-text-muted)", margin: "6px 0 2px" } }, t);

    const wheel = h("div", {
      ref: wrapRef,
      onPointerDown: onWheelDown,
      onPointerMove: onWheelMove,
      onPointerUp: onWheelUp,
      onPointerCancel: onWheelUp,
      style: {
        position: "relative", width: SIZE + "px", height: SIZE + "px",
        margin: "2px auto 4px", borderRadius: "50%", cursor: "crosshair",
        touchAction: "none", boxShadow: "inset 0 0 0 1px var(--color-border)",
      },
    }, [
      h("canvas", { key: "cv", ref: canvasRef, width: SIZE, height: SIZE,
        style: { width: SIZE + "px", height: SIZE + "px", borderRadius: "50%", display: "block" } }),
      ...markers,
    ]);

    const controls = sel >= 0 ? h("div", { key: "ctrls" }, [
      h("div", { key: "hd", style: { display: "flex", justifyContent: "space-between",
        alignItems: "center", marginTop: "4px" } }, [
        h("span", { key: "t", style: { fontSize: "11px", color: "var(--color-text-secondary)" } },
          `Point ${active.indexOf(sel) + 1} of ${active.length}`),
        h("button", { key: "rm", onClick: removeSel,
          style: { fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.05em",
            padding: "1px 6px", borderRadius: "3px", cursor: "pointer",
            border: "1px solid var(--color-border-subtle)", background: "transparent",
            color: "var(--color-text-secondary)" } }, "Remove"),
      ]),
      label("Selection"),
      h(Slider, { key: "range", label: "Range", value: Math.round((a[2] / 1.5) * 100),
        min: 5, max: 100, defaultValue: 40, onChange: (v) => setA(2, (v / 100) * 1.5), onCommit: commit }),
      h(Slider, { key: "soft", label: "Smooth", value: Math.round(a[3] * 100),
        min: 2, max: 100, defaultValue: 50, onChange: (v) => setA(3, v / 100), onCommit: commit }),
      label("Adjust"),
      h(Slider, { key: "hue", label: "Hue", value: Math.round(b[0] * 100),
        min: -100, max: 100, defaultValue: 0, onChange: (v) => setB(0, v / 100), onCommit: commit }),
      h(Slider, { key: "sat", label: "Saturation", value: Math.round(b[1] * 100),
        min: -100, max: 100, defaultValue: 0, onChange: (v) => setB(1, v / 100), onCommit: commit }),
      h(Slider, { key: "lum", label: "Luminance", value: Math.round(b[2] * 100),
        min: -100, max: 100, defaultValue: 0, onChange: (v) => setB(2, v / 100), onCommit: commit }),
    ]) : h("div", { key: "empty", style: { fontSize: "11px", color: "var(--color-text-muted)",
      textAlign: "center", padding: "6px 4px" } },
      "Click the wheel to add a colour point, then reshape its hue, saturation and luminance.");

    const countDiv = h("div", { style: { fontSize: "10px", color: "var(--color-text-muted)",
      textAlign: "center", marginBottom: "2px" } }, `${active.length}/${MAX_POINTS} points`);

    return h(Panel, { title: "Color Editor" }, wheel, countDiv, controls);
  }

  api.registerPanel({
    id: PANEL_ID,
    title: "Color Editor",
    component: ColorEditorPanel,
    defaultDock: { module: "develop", direction: "right", order: 12, width: 280, height: 360 },
    onReset: clearAll,
  });
}

export function deactivate() {
  // Contributions are swept by the host on disable; just drop our own listeners.
  if (_retimer) { clearTimeout(_retimer); _retimer = null; }
  if (_settingsUnsub) { _settingsUnsub(); _settingsUnsub = null; }
}
