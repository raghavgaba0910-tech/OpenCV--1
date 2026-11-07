# Fire & Water AR (Realistic Elemental Hand FX)

A real-time webcam augmented reality application that tracks hands using MediaPipe and wraps them in realistic, high-impact elemental **Fire** and **Water** effects.

---

## What's New & Upgraded

- **Full Anatomical Hand Coverage:** Replaced crude skeletons and flat boxy hulls with contoured, continuous anatomical skin. Every finger, knuckle, and palm is individually coated in elemental textures that bend and move naturally with your hand.
- **Realistic Volumetric Fire:**
  - Magma skin with procedural churning lava veins.
  - White-hot core bones and palm thermal incandescence.
  - Multi-tier dynamic flame tongues billowing upward from fingertips and knuckles.
  - Fire aerodynamics: flames trail behind the hand when moved rapidly.
  - Flying ember sparks with thermal buoyancy and erratic turbulent dance.
- **Realistic Volumetric Water:**
  - Translucent liquid hydro-skin with real-time procedural caustics and wave ripple patterns.
  - Specular Fresnel rim highlights along finger contours.
  - 3D rotating orbital water vortex rings circling around the palm and wrist.
  - 3D liquid droplets with specular highlights that fling dynamically with centrifugal force.
- **Accurate Orientation-Invariant Gesture Recognition:**
  - 3D landmark Euclidean distance ratios and joint straightness metrics that work accurately at any angle (upside-down, tilted, sideways).
  - Multi-frame temporal voting window to eliminate gesture flickering.
- **Elemental Fusion & Dual-Hand Magic:**
  - **Fire + Water:** Boiling steam clouds, vapor condensation shockwaves, and electrical sizzle arcs.
  - **Fire + Fire:** Solar flare inferno with central plasma core.
  - **Water + Water:** Double hydro maelstrom vortex.
- **Supercharged Fist Mode:** Making a fist (0 fingers) condenses the elemental power into a pressurized, supercharged state.
- **Atmospheric Bloom:** Multi-pass Gaussian blur glow that lights up the surroundings and ambient frame.

---

## Gestures & Controls

| Gesture | Element | Description |
|---|---|---|
| **1 Finger Up** | 🔥 **Fire** | Engulfs hand in molten lava skin, rising flame plumes, and flying embers |
| **2+ Fingers Up** | 💧 **Water** | Wraps hand in glistening water caustics, orbital vortex rings, and splashing droplets |
| **0 Fingers (Fist)** | ⚡ **Supercharge** | Locks and condenses current element into a pressurized, high-energy fist |
| **Bring Hands Together** | 💥 **Fusion** | Creates steam explosions, solar flare bridges, or hydro maelstroms |
| **`H` Key** | 🖥️ **Toggle HUD** | Hides or shows the top status bar overlay |
| **`C` Key** | 🦴 **Skeleton** | Toggles subtle skeletal guide overlay (default OFF for realistic look) |
| **`R` Key** | 🔄 **Reset** | Clears active particles |
| **`Q` Key / `ESC`** | ❌ **Exit** | Quits the application and releases the webcam |

---

## How to Run

1. Open PowerShell / Command Prompt in this folder.
2. Run:
   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```
   *(Or double-click `run.bat`)*
