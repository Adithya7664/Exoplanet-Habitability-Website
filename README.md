# 🪐 PlanetScope — Exoplanet Habitability Explorer

An interactive web app for exploring NASA's exoplanet archive, predicting planetary habitability with a machine learning ensemble, and generating AI-powered terraforming roadmaps.

**Live demo:** [exoplanet-habitability-website-sthvjpfpl9ucwqvpdedmdd.streamlit.app](https://exoplanet-habitability-website-sthvjpfpl9ucwqvpdedmdd.streamlit.app/)

---

## What it does

PlanetScope lets you explore thousands of confirmed exoplanets through five interactive pages:

| Page | Description |
|------|-------------|
| 🏠 **Home** | Dashboard with top 10 most Earth-like planets, ESI distribution curve, polar orbital map, and nearest Earth-like planets chart |
| 🔭 **Explorer** | Filter planets by ESI, distance, stellar temperature, and habitable zone; visualise results as a 3D density surface and habitability funnel |
| 🪐 **Planet Details** | Three-level selector (Cluster → Star System → Planet); real-time 3D planet render, ESI radar chart, and SHAP feature breakdown |
| 🛸 **Terraforming AI** | Pick any planet and Claude generates a structured terraforming roadmap — difficulty rating, timeline, phased plan, and required technologies |
| ✏️ **Custom Designer** | Dial in your own planet parameters and get an instant ML habitability prediction, radar chart, 3D render, and nearest real-planet matches |

---

## Features

### Machine Learning
- **Ensemble model** combining Random Forest, XGBoost, and a Neural Network (MLP) trained on engineered features from the NASA Exoplanet Archive
- **Earth Similarity Index (ESI)** computed from radius, temperature, mass, and insolation flux
- **SHAP explanations** showing which physical features push habitability up or down for any planet

### 3D Planet Rendering
- Real-time WebGL rendering via Three.js embedded in the browser — no plugin required
- **Rocky planets:** procedural terrain (fractal noise + craters), oceans, polar ice caps, cloud layer, and atmospheric glow — all physically derived from the planet's actual temperature, ESI, stellar type, and age
- **Gas giants:** multi-frequency banded appearance with natural width variation, temperature-based colour palettes (ice blue → warm orange → deep crimson), storm spots, and diffuse lighting
- Fully interactive: drag to rotate, scroll to zoom, auto-rotates

### AI Terraforming
- Powered by the Claude API (Anthropic)
- Returns structured output: difficulty rating, estimated timeline, key challenges, and a phased engineering plan with Terragenesis-style mechanics

---

## Tech stack

| Layer | Libraries |
|-------|-----------|
| UI | Streamlit, Plotly |
| ML | scikit-learn, XGBoost, SHAP, NumPy, pandas, SciPy |
| AI | Anthropic Claude API |
| 3D rendering | Three.js (WebGL, GLSL shaders) |
| Data | NASA Exoplanet Archive |

---

## Project structure

```
├── app.py                           # Main Streamlit app (all 5 pages)
├── train.py                         # Model training script
├── requirements.txt
├── .env                             # API keys (not committed)
├── data/
│   └── exoplanets_engineered.csv   # Feature-engineered planet dataset
├── models/
│   └── ensemble.joblib             # Trained model weights
└── src/
    ├── features.py                  # ESI computation and feature engineering
    ├── model.py                     # Ensemble model definition, SHAP explainer
    ├── renderer.py                  # WebGL / Three.js planet renderer
    └── terraformer.py               # Claude API terraforming plan generator
```

---

## Running locally

### 1. Clone the repo

```bash
git clone https://github.com/Adithya7664/Exoplanet-Habitability-Website.git
cd Exoplanet-Habitability-Website
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

The Terraforming AI page requires this. All other pages work without it.

### 4. Train the model (first time only)

```bash
python train.py
```

This reads `data/exoplanets_engineered.csv`, trains the ensemble, and saves `models/ensemble.joblib`.

### 5. Run the app

```bash
streamlit run app.py
```

---

## Data

The dataset is derived from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) with additional engineered features including ESI components, insolation flux, stellar luminosity proxy, and habitable zone flag.

---

## Acknowledgements

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) for the planetary data
- [Anthropic](https://anthropic.com) for the Claude API
- [Three.js](https://threejs.org) for WebGL rendering
