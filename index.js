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
const SLOT_ID = "color2d-editor.picker-overlay";
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

// `size` is the canvas's device-pixel backing-store size (CSS size × dpr), so the
// wheel stays crisp on HiDPI. The rim alpha feathers over ~1.5px centred on the
// edge, so the circle reads as smooth/round instead of a jagged pixel staircase.
function drawWheel(canvas, size) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const img = ctx.createImageData(size, size);
  const d = img.data;
  const r0 = size / 2;
  const aa = 1.5 / r0; // anti-aliased rim width, in normalised-radius units
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x - r0 + 0.5) / r0;
      const dy = (y - r0 + 0.5) / r0;
      const rad = Math.hypot(dx, dy);
      const idx = (y * size + x) * 4;
      if (rad > 1 + aa) { d[idx + 3] = 0; continue; }
      let ang = Math.atan2(-dy, dx);
      if (ang < 0) ang += TWO_PI;
      const [rr, gg, bb] = hsvToRgb(ang / TWO_PI, Math.min(rad, 1), 1);
      const alpha = Math.min(1, Math.max(0, (1 + aa - rad) / (2 * aa)));
      d[idx] = rr; d[idx + 1] = gg; d[idx + 2] = bb;
      d[idx + 3] = Math.round(alpha * 255);
    }
  }
  ctx.putImageData(img, 0, 0);
}

// sRGB (0..1) -> HSL. Matches the shader's rgb2hsl so a colour picked off the
// image lands at the same wheel position the stage selects.
function rgb2hsl(r, g, b) {
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  let h = 0;
  let s = 0;
  const l = (mx + mn) / 2;
  const d = mx - mn;
  if (d > 1e-5) {
    s = l < 0.5 ? d / (mx + mn) : d / (2 - mx - mn);
    if (mx === r) h = (g - b) / d + (g < b ? 6 : 0);
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h /= 6;
  }
  return [h, s, l];
}

// Minimal shared store so the panel's Pick button and the canvas overlay agree on
// picker mode + the selected point (React is external; no zustand to import).
function makeStore(initial) {
  let state = initial;
  const subs = new Set();
  return {
    get: () => state,
    set: (patch) => { state = { ...state, ...patch }; subs.forEach((f) => f()); },
    subscribe: (f) => { subs.add(f); return () => subs.delete(f); },
  };
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

  // ---- shared UI state (picker mode + selected point) ----
  const ui = makeStore({ picking: false, selected: -1 });
  function usePicker() {
    const [, bump] = React.useState(0);
    React.useEffect(() => ui.subscribe(() => bump((n) => n + 1)), []);
    return ui.get();
  }

  // Drop a point at a chromaticity-disc position (hue turns, saturation), adopting
  // the user's default selection style. Commits immediately (one undo step) and
  // selects it. When all slots are full, retarget the selected point instead.
  function addPointAt(hue, sat) {
    const bag = useDevelopStore.getState().paramBag;
    const range = (api.settings.get("defaultRange", DEFAULT_RANGE) / 100) * 1.5;
    const soft = api.settings.get("defaultSmooth", DEFAULT_SMOOTH) / 100;
    let free = -1;
    for (let i = 0; i < MAX_POINTS; i++) if (readB(bag, i)[3] < 0.5) { free = i; break; }
    if (free >= 0) {
      writeSlots({ [KA(free)]: [hue, sat, range, soft], [KB(free)]: [0, 0, 0, 1] });
      commit();
      ui.set({ selected: free });
      return;
    }
    const selPrev = ui.get().selected;
    if (selPrev >= 0 && readB(bag, selPrev)[3] >= 0.5) {
      const a = readA(bag, selPrev);
      writeSlots({ [KA(selPrev)]: [hue, sat, a[2], a[3]] });
      commit();
    }
  }

  // Sample the displayed colour under a canvas point and add a selection point.
  // The captured frame is aligned 1:1 with the displayed canvas, so the click is
  // mapped through the overlay's `rect` (where the displayed pixels sit). clientWidth
  // vs getBoundingClientRect width yields the interface-scale factor, so the map
  // holds under Windows DPI + interface zoom (see pointer-coords memory).
  async function sampleColorAt(clientX, clientY, el, ovr) {
    // Whole body guarded: a sampling failure (no overlay rect, capture/decode
    // error, OffscreenCanvas/getImageData throwing) must never crash the canvas.
    try {
      const rect = ovr.rect;
      if (!rect || !el) return;
      const obr = el.getBoundingClientRect();
      const sx = obr.width ? el.clientWidth / obr.width : 1;
      const sy = obr.height ? el.clientHeight / obr.height : 1;
      const lx = (clientX - obr.left) * sx;
      const ly = (clientY - obr.top) * sy;
      const nx = (lx - rect.x) / rect.w;
      const ny = (ly - rect.y) / rect.h;
      if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return; // outside the image
      const st = useDevelopStore.getState();
      const bmp = await api.develop.captureFrame(st.previewParams || st.params);
      const w = bmp.width;
      const hgt = bmp.height;
      if (!w || !hgt) return;
      const oc = new OffscreenCanvas(w, hgt);
      const ctx = oc.getContext("2d");
      ctx.drawImage(bmp, 0, 0);
      if (bmp.close) bmp.close();
      const px = Math.max(0, Math.min(w - 1, Math.round(nx * w)));
      const py = Math.max(0, Math.min(hgt - 1, Math.round(ny * hgt)));
      const d = ctx.getImageData(px, py, 1, 1).data;
      const [hh, ss] = rgb2hsl(d[0] / 255, d[1] / 255, d[2] / 255);
      if (ss < 0.04) return; // near-neutral: no meaningful hue to select
      addPointAt(hh, ss);
    } catch (err) {
      console.warn("[color2d-editor] colour pick failed:", err);
    }
  }

  // Canvas overlay active only in pick mode. The global Ctrl/⌘/Space passthrough
  // (core) forces this click-through while a gesture key is held, so pan/zoom keep
  // working; here we just capture a plain click to sample.
  function PickerOverlay() {
    const { picking } = usePicker();
    const ovr = api.develop.useDevelopOverlay();
    const elRef = React.useRef(null);
    React.useEffect(() => {
      if (!picking) return;
      const onKey = (e) => { if (e.key === "Escape") ui.set({ picking: false }); };
      window.addEventListener("keydown", onKey);
      return () => window.removeEventListener("keydown", onKey);
    }, [picking]);
    if (!picking) return null;
    return h("div", {
      ref: elRef,
      onPointerDown: (e) => {
        if (e.button !== 0 || e.ctrlKey || e.metaKey) return;
        e.preventDefault();
        void sampleColorAt(e.clientX, e.clientY, elRef.current, ovr);
      },
      style: {
        position: "absolute", inset: 0, pointerEvents: "auto",
        // The app's shared colour-picker cursor (honours any Custom Cursors theme),
        // matching the built-in HSL picker. Guarded: api.cursors.resolve was added
        // in a later core build, so fall back to crosshair on older ones (calling
        // an undefined resolve here would throw mid-render and crash the canvas).
        cursor: api.cursors && api.cursors.resolve ? api.cursors.resolve("pick") : "crosshair",
      },
    });
  }

  // ---- panel ----
  function ColorEditorPanel() {
    const paramBag = useDevelopStore((s) => s.paramBag);
    const { selected, picking } = usePicker();
    const canvasRef = React.useRef(null);
    const wrapRef = React.useRef(null);
    const dragRef = React.useRef(null);
    const SIZE = 188; // CSS size; the canvas backing store is SIZE × dpr for crispness
    const dpr = Math.min((typeof window !== "undefined" && window.devicePixelRatio) || 1, 3);
    const PIX = Math.round(SIZE * dpr);

    React.useEffect(() => {
      if (canvasRef.current) drawWheel(canvasRef.current, PIX);
    }, [PIX]);

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
      ui.set({ selected: hit });
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

    // Double-click a point to remove it (no Remove button — matches the canvas
    // "double-click to delete" convention).
    const onWheelDoubleClick = (e) => {
      const r = wrapRef.current.getBoundingClientRect();
      for (const i of active) {
        const p = posOf(i);
        if (Math.hypot(e.clientX - r.left - p.x, e.clientY - r.top - p.y) <= 11) {
          dragRef.current = null; // cancel any pending drag on the removed slot
          writeSlots({ [KA(i)]: ZERO4, [KB(i)]: ZERO4 });
          commit();
          ui.set({ selected: -1 });
          return;
        }
      }
    };

    // Slider helpers for the selected point.
    const a = sel >= 0 ? readA(paramBag, sel) : ZERO4;
    const b = sel >= 0 ? readB(paramBag, sel) : ZERO4;
    const setA = (idx, val) => { const na = a.slice(); na[idx] = val; writeSlots({ [KA(sel)]: na }); };
    const setB = (idx, val) => { const nb = b.slice(); nb[idx] = val; writeSlots({ [KB(sel)]: nb }); };

    // Selected point reads boldly: bigger, a solid white ring, and a white glow
    // halo over a dark outline so it stands out against any wheel colour.
    const markers = active.map((i) => {
      const p = posOf(i);
      const isSel = i === sel;
      const sz = isSel ? 18 : 12;
      return h("div", {
        key: i,
        style: {
          position: "absolute", left: p.x + "px", top: p.y + "px",
          width: sz + "px", height: sz + "px",
          marginLeft: -sz / 2 + "px", marginTop: -sz / 2 + "px",
          borderRadius: "50%", boxSizing: "border-box", background: "transparent",
          border: isSel ? "3px solid #fff" : "2px solid rgba(255,255,255,0.85)",
          boxShadow: isSel
            ? "0 0 0 2px rgba(0,0,0,0.75), 0 0 9px 2px rgba(255,255,255,0.85)"
            : "0 0 0 1.5px rgba(0,0,0,0.6)",
          pointerEvents: "none",
          zIndex: isSel ? 2 : 1,
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
      onDoubleClick: onWheelDoubleClick,
      style: {
        position: "relative", width: SIZE + "px", height: SIZE + "px",
        margin: "2px auto 4px", borderRadius: "50%", cursor: "crosshair",
        touchAction: "none", boxShadow: "inset 0 0 0 1px var(--color-border)",
      },
    }, [
      h("canvas", { key: "cv", ref: canvasRef, width: PIX, height: PIX,
        style: { width: SIZE + "px", height: SIZE + "px", borderRadius: "50%", display: "block" } }),
      ...markers,
    ]);

    const controls = sel >= 0 ? h("div", { key: "ctrls" }, [
      h("div", { key: "hd", style: { display: "flex", justifyContent: "space-between",
        alignItems: "center", marginTop: "4px" } }, [
        h("span", { key: "t", style: { fontSize: "11px", color: "var(--color-text-secondary)" } },
          `Point ${active.indexOf(sel) + 1} of ${active.length}`),
        h("span", { key: "hint", style: { fontSize: "9px", textTransform: "uppercase",
          letterSpacing: "0.04em", color: "var(--color-text-muted)" } }, "Double-click to remove"),
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

    const path = (d) => h("path", { d });
    const eyedropper = h("svg", {
      width: 11, height: 11, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
      strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true,
    }, path("m2 22 1-1h3l9-9"), path("M3 21v-3l9-9"),
       path("m15 6 3.4-3.4a2.1 2.1 0 1 1 3 3L18 9l.4.4a2.1 2.1 0 1 1-3 3l-3.8-3.8a2.1 2.1 0 1 1 3-3l.4.4Z"));
    const pickerBar = h("div", { style: { display: "flex", justifyContent: "center", marginBottom: "4px" } },
      h("button", {
        onClick: () => ui.set({ picking: !picking }),
        "aria-pressed": picking,
        title: "Pick a colour from the image to drop a selection point (Esc or click again to stop)",
        style: {
          display: "inline-flex", alignItems: "center", gap: "5px",
          fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.06em",
          padding: "2px 9px", borderRadius: "4px", cursor: "pointer",
          border: "1px solid " + (picking ? "var(--color-accent)" : "var(--color-border-subtle)"),
          background: picking ? "var(--color-accent)" : "transparent",
          color: picking ? "#fff" : "var(--color-text-secondary)",
        },
      }, eyedropper, "Pick from image"));

    return h(Panel, { title: "Color Editor" }, pickerBar, wheel, countDiv, controls);
  }

  api.registerPanel({
    id: PANEL_ID,
    title: "Color Editor",
    component: ColorEditorPanel,
    defaultDock: { module: "develop", direction: "right", order: 12, width: 280, height: 360 },
    onReset: clearAll,
  });

  // On-image eyedropper: a develop-canvas overlay, live only while Pick is armed.
  api.registerSlot({
    id: SLOT_ID,
    slot: "develop-canvas-overlay",
    component: PickerOverlay,
    order: 60,
  });
}

export function deactivate() {
  // Contributions are swept by the host on disable; just drop our own listeners.
  if (_retimer) { clearTimeout(_retimer); _retimer = null; }
  if (_settingsUnsub) { _settingsUnsub(); _settingsUnsub = null; }
}
