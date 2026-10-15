// Tests for the Continuous 2D Color Editor's colour wheel. Plain Node, no
// dependencies:
//
//   node --test        (npm test runs the same thing)
//
// The wheel is covered two ways: its pure geometry and drawing directly, and the
// real panel through a stand-in host API. The panel's hooks are inert there, so a
// render is a plain call and its pointer handlers are invoked with events shaped
// like the browser's.

import test from "node:test";
import assert from "node:assert/strict";
import {
  activate,
  deactivate,
  drawWheel,
  hitWheelPoint,
  rgb2hsl,
  wheelMarkerPos,
  wheelPointerPos,
} from "./index.js";

const SIZE = 188; // the wheel box's layout size, as the panel styles it
const ZOOMS = [1, 1.2, 0.8]; // Interface scale at 100%, 120% and 80%

const near = (actual, expected, eps, msg) =>
  assert.ok(Math.abs(actual - expected) <= eps, `${msg}: ${actual} is not within ${eps} of ${expected}`);

// The wheel box and a pointer over it as the browser reports them when Interface
// scale puts CSS `zoom` on <body>: client coordinates and getBoundingClientRect()
// are zoomed visual pixels, `zoom` times the layout pixels the markers sit in.
function underZoom(zoom, size = SIZE) {
  const rect = { left: 37 * zoom, top: 211 * zoom, width: size * zoom, height: size * zoom };
  const pointerAt = (p) => [rect.left + p.x * zoom, rect.top + p.y * zoom]; // p in layout px
  return { rect, pointerAt };
}

// Saved points as the panel reads them: A = (hueTurns, sat, range, soft).
const POINTS = [
  { slot: 0, a: [0.0, 0.9, 0.6, 0.5] },
  { slot: 3, a: [0.25, 0.5, 0.6, 0.5] },
  { slot: 6, a: [0.6, 0.8, 0.6, 0.5] },
];

// ---------------------------------------------------------------------------
// Hit-test geometry
// ---------------------------------------------------------------------------

test("wheelMarkerPos puts hue anticlockwise from the right and saturation along the radius", () => {
  const cases = [
    [[0, 1], { x: 188, y: 94 }],
    [[0.25, 1], { x: 94, y: 0 }],
    [[0.5, 0.5], { x: 47, y: 94 }],
    [[0.75, 1], { x: 94, y: 188 }],
    [[0.3, 0], { x: 94, y: 94 }],
  ];
  for (const [a, want] of cases) {
    const got = wheelMarkerPos(a, SIZE);
    near(got.x, want.x, 1e-9, `x of ${a}`);
    near(got.y, want.y, 1e-9, `y of ${a}`);
  }
});

test("wheelPointerPos divides the visual offset by the rendered/layout size ratio", () => {
  // A 188px box at zoom 1.2 reports 225.6px, so 120 visual px in is 100 layout px.
  const zoomed = wheelPointerPos({ left: 100, top: 50, width: 225.6, height: 225.6 }, SIZE, 220, 110);
  near(zoomed.x, 100, 1e-9, "x at 120%");
  near(zoomed.y, 50, 1e-9, "y at 120%");
  const plain = wheelPointerPos({ left: 100, top: 50, width: 188, height: 188 }, SIZE, 130, 90);
  near(plain.x, 30, 1e-9, "x at 100%");
  near(plain.y, 40, 1e-9, "y at 100%");
});

for (const zoom of ZOOMS) {
  test(`a point under the pointer is found at zoom ${zoom}`, () => {
    const { rect, pointerAt } = underZoom(zoom);
    for (const { slot, a } of POINTS) {
      const [x, y] = pointerAt(wheelMarkerPos(a, SIZE));
      assert.equal(hitWheelPoint(POINTS, rect, SIZE, x, y), slot, `slot ${slot}`);
    }
  });

  test(`the hit radius is 11 layout pixels at zoom ${zoom}`, () => {
    const { rect, pointerAt } = underZoom(zoom);
    const p = wheelMarkerPos(POINTS[1].a, SIZE);
    const along = (dx) => hitWheelPoint(POINTS, rect, SIZE, ...pointerAt({ x: p.x + dx, y: p.y }));
    assert.equal(along(10), 3, "10px right of the marker");
    assert.equal(along(-10), 3, "10px left of the marker");
    assert.equal(along(12), -1, "12px right of the marker");
    assert.equal(along(-12), -1, "12px left of the marker");
  });

  test(`a pointer away from every point is a miss at zoom ${zoom}`, () => {
    const { rect, pointerAt } = underZoom(zoom);
    for (const spot of [{ x: 94, y: 94 }, { x: 60, y: 160 }, { x: 150, y: 150 }, { x: 10, y: 10 }]) {
      assert.equal(hitWheelPoint(POINTS, rect, SIZE, ...pointerAt(spot)), -1, `${spot.x},${spot.y}`);
    }
    assert.equal(hitWheelPoint([], rect, SIZE, ...pointerAt({ x: 94, y: 94 })), -1, "no points");
  });

  if (zoom !== 1) {
    test(`a point's unscaled hit zone is a miss at zoom ${zoom}`, () => {
      // Reading a layout position as a visual offset puts it at layout position
      // p / zoom: the spot the unscaled comparison used to accept.
      const { rect, pointerAt } = underZoom(zoom);
      for (const { slot, a } of POINTS) {
        const p = wheelMarkerPos(a, SIZE);
        const [x, y] = pointerAt({ x: p.x / zoom, y: p.y / zoom });
        assert.equal(hitWheelPoint(POINTS, rect, SIZE, x, y), -1, `slot ${slot}`);
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Wheel colours
// ---------------------------------------------------------------------------

// Draw the wheel the way the panel does, into a fake canvas, and read pixels back.
function drawnWheel(size) {
  let image = null;
  const ctx = {
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: (img) => { image = img; },
  };
  drawWheel({ getContext: () => ctx }, size);
  return (x, y) => {
    const i = (Math.floor(y) * size + Math.floor(x)) * 4;
    return [...image.data.slice(i, i + 4)];
  };
}

test("the colour drawn under a point is the colour that point selects", () => {
  const pixel = drawnWheel(SIZE);
  for (const hue of [0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85]) {
    for (const sat of [0.3, 0.5, 0.7, 0.9]) {
      const at = wheelMarkerPos([hue, sat], SIZE);
      const [r, g, b] = pixel(at.x, at.y);
      const [h, s, l] = rgb2hsl(r / 255, g / 255, b / 255);
      const read = [h, s, l].map((v) => v.toFixed(3)).join(", ");
      const label = `hue ${hue}, sat ${sat}: drawn colour reads as HSL(${read})`;
      near(s, sat, 0.03, label);
      near(l, 0.5, 0.03, label);
      const dh = Math.abs(h - hue);
      near(Math.min(dh, 1 - dh), 0, 0.02, label);
    }
  }
});

// ---------------------------------------------------------------------------
// The real panel, driven through a stand-in host API
// ---------------------------------------------------------------------------

// The saved format: one pair of vec4s per slot in the develop param bag.
const KA = (i) => `color2d-editor.color.uCP${i}A`;
const KB = (i) => `color2d-editor.color.uCP${i}B`;

function* walk(node) {
  yield node;
  if (Array.isArray(node)) for (const n of node) yield* walk(n);
  else if (node && typeof node === "object") yield* walk(node.children);
}
const isElement = (n) => n !== null && typeof n === "object" && !Array.isArray(n);

// Activate the extension against the stand-in API. `saved` lists [hueTurns, sat]
// points already in the photo's param bag.
function openEditor(saved = []) {
  const bag = {};
  saved.forEach(([hue, sat], i) => {
    bag[KA(i)] = [hue, sat, 0.6, 0.5];
    bag[KB(i)] = [0, 0, 0, 1];
  });
  const writes = [];
  const commits = [];
  const state = {
    paramBag: bag,
    setDynParams: (patch) => { writes.push(patch); Object.assign(bag, patch); },
    commitEdit: async (label) => { commits.push(label); },
  };
  const registered = {};
  activate({
    react: {
      createElement: (type, props, ...children) => ({ type, props: props ?? {}, children }),
      useRef: (current) => ({ current }),
      useState: (value) => [value, () => {}],
      useEffect: () => {},
    },
    components: { Panel: "Panel", Slider: "Slider" },
    stores: { useDevelopStore: Object.assign((select) => select(state), { getState: () => state }) },
    settings: { get: (_key, fallback) => fallback, onChange: () => () => {} },
    registerProcessingStage: (stage) => { registered.stage = stage; },
    registerSettings: (settings) => { registered.settings = settings; },
    registerPanel: (panel) => { registered.panel = panel; },
    registerSlot: (slot) => { registered.slot = slot; },
  });

  // Render the panel once with the wheel box placed under `zoom`.
  function render(zoom) {
    const tree = registered.panel.component();
    const wheel = [...walk(tree)].filter(isElement).find((n) => n.props.onDoubleClick);
    const { rect, pointerAt } = underZoom(zoom, parseFloat(wheel.props.style.width));
    wheel.props.ref.current = { getBoundingClientRect: () => rect };
    return {
      size: parseFloat(wheel.props.style.width),
      pointerAt,
      // Layout position of each drawn marker, in slot order.
      markers: [...walk(wheel.children)]
        .filter((n) => isElement(n) && n.type === "div")
        .map((m) => ({ x: parseFloat(m.props.style.left), y: parseFloat(m.props.style.top) })),
      texts: [...walk(tree)].filter((n) => typeof n === "string"),
      press: ([clientX, clientY]) => wheel.props.onPointerDown({
        button: 0, pointerId: 1, clientX, clientY,
        preventDefault() {}, currentTarget: { setPointerCapture() {} },
      }),
      doubleClick: ([clientX, clientY]) => wheel.props.onDoubleClick({ clientX, clientY }),
    };
  }

  return { bag, writes, commits, registered, render, close: () => deactivate() };
}

const SAVED = [[0.0, 0.9], [0.25, 0.5], [0.6, 0.8]];

for (const zoom of ZOOMS) {
  test(`pressing a point selects it and adds no second one at zoom ${zoom}`, () => {
    const editor = openEditor(SAVED);
    const view = editor.render(zoom);
    assert.equal(view.markers.length, 3);
    view.press(view.pointerAt(view.markers[1]));
    assert.equal(editor.writes.length, 0, "a press on a marker must not add a point");
    assert.ok(editor.render(zoom).texts.includes("Point 2 of 3"), "the pressed point is selected");
    editor.close();
  });

  test(`double-clicking a point removes it at zoom ${zoom}`, () => {
    const editor = openEditor(SAVED);
    const view = editor.render(zoom);
    view.doubleClick(view.pointerAt(view.markers[1]));
    assert.deepEqual(editor.bag[KB(1)], [0, 0, 0, 0], "the double-clicked point is cleared");
    assert.deepEqual(editor.bag[KB(0)], [0, 0, 0, 1], "the first point stays");
    assert.deepEqual(editor.bag[KB(2)], [0, 0, 0, 1], "the last point stays");
    assert.equal(editor.commits.length, 1, "one undo step");
    editor.close();
  });

  test(`double-clicking away from every point removes nothing at zoom ${zoom}`, () => {
    const editor = openEditor(SAVED);
    const view = editor.render(zoom);
    const spots = [{ x: 94, y: 94 }];
    // Where the unscaled comparison used to find a point.
    if (zoom !== 1) spots.push(...view.markers.map((m) => ({ x: m.x / zoom, y: m.y / zoom })));
    for (const spot of spots) view.doubleClick(view.pointerAt(spot));
    assert.equal(editor.writes.length, 0, "nothing written");
    assert.equal(editor.commits.length, 0, "nothing committed");
    editor.close();
  });

  test(`pressing empty wheel drops a point under the pointer at zoom ${zoom}`, () => {
    const editor = openEditor([[0.0, 0.9]]);
    const view = editor.render(zoom);
    const spot = { x: 60, y: 160 }; // layout px, clear of the one marker
    view.press(view.pointerAt(spot));
    assert.equal(editor.writes.length, 1, "one new point");
    const placed = wheelMarkerPos(editor.bag[KA(1)], view.size);
    near(placed.x, spot.x, 1e-6, "marker x");
    near(placed.y, spot.y, 1e-6, "marker y");
    editor.close();
  });
}

test("activate registers the stage, settings, panel and overlay slot the app expects", () => {
  const editor = openEditor();
  const { stage, settings, panel, slot } = editor.registered;
  assert.equal(stage.id, "color2d-editor.color");
  assert.equal(stage.phase, "display-adjust");
  const keys = Array.from({ length: 8 }, (_, i) => [`uCP${i}A`, `uCP${i}B`]).flat().concat("uCPCount");
  assert.deepEqual(stage.uniforms.map((u) => u.key), keys);
  assert.equal(typeof settings.title, "string");
  assert.equal(panel.id, "color2d-editor.panel");
  assert.equal(slot.slot, "develop-canvas-overlay");
  assert.equal(typeof deactivate, "function");
  editor.close();
});
