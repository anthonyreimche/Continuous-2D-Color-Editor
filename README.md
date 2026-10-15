# Continuous 2D Color Editor

A continuous colour editor for [Safelight](https://github.com/anthonyreimche/Safelight) — the Capture One–style alternative to a fixed 8‑band HSL mixer.

Instead of eight preset hue buckets, you drop **points anywhere on a hue/saturation wheel**. Each point selects a smooth *2D* region of colour — a disc on the chromaticity plane, with its own size and softness — and reshapes that region's **hue, saturation and luminance**. Where points overlap, their adjustments add up. It's the right tool when the colour you want to push sits between the stock bands, or when you want to isolate "the saturated blues" without touching the pale ones.

## Using it

- Open the **Color Editor** panel in Develop (right rail).
- **Click the wheel** to drop a colour point. Its angle is the hue, its distance from the centre is the saturation — so a point near the rim targets vivid colours, one near the centre targets muted ones.
- **Pick from image**: click the eyedropper button, then click anywhere on the photo to drop a point at exactly that colour. It stays armed for multiple picks — press Esc or click the button again to stop. Hold Ctrl/⌘ (or Space) to pan/zoom while picking.
- **Drag a point** to retarget it; **click a point** to select it (the selected point is drawn larger with a bright halo). **Double-click a point** to remove it.
- With a point selected, use the sliders:
  - **Range** — how wide a colour region the point covers.
  - **Smooth** — how soft the selection's edge is.
  - **Hue / Saturation / Luminance** — the adjustment applied to the selected region.
- The panel header's reset clears all points.

Up to 8 points per photo. Everything saves with the edit and undoes/redoes like any slider.

## Colour science

Selection is a smooth radial falloff in the hue–saturation disc, so there are no hard band edges. The adjustments use the same model as Safelight's built‑in HSL mixer: near‑neutral pixels are protected (greys aren't tinted), hue rotates in HSL, saturation is anchored to true luminance, and luminance is a hue‑preserving gain. The work runs per‑pixel on the GPU as a registered processing stage in the display‑adjust phase, so it applies live in Develop and on export.

## Installing

This extension is a single self‑contained ESM bundle — no build step.

- **From the store:** publish this repo to GitHub, tag it with the `safelight-extension` topic, and install it from Safelight's Extensions window.
- **Locally (for development):** enable **Developer Tools** in the Extensions manager and point its dev folder at this repository; Safelight will load `index.js` directly.

## License

MIT © 2026 Anthony Reimche
