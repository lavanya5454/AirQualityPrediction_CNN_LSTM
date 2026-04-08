# AirSense - PM2.5 Predictor

A Streamlit web app for interactive PM2.5 forecasting and air-quality visualization.

## Features

- Interactive environmental inputs (temperature, dew point, pressure, wind, rain/snow, date/time)
- Simulated current PM2.5 prediction + 24-hour forecast
- AQI-style category banner (`Good`, `Moderate`, `Sensitive`, etc.)
- Historical 48-hour synthetic data visualizations
- Model-insight dashboard (architecture view, loss curves, attention heatmap)

## Tech Stack

- Python 3.9+
- Streamlit
- NumPy
- Pandas
- Plotly

## Project Structure

```text
AirQualityPred/
  AirQualityPred.ipynb
  app.py
  best_aq_model.h5
  feature_scaler.pkl
```
- `AirQualityPred.ipynb`: Jupyter notebook for data exploration, model training, and artifact generation
- `app.py`: Main Streamlit application and simulator logic
- `best_aq_model.h5`: Trained model artifact (not auto-loaded in current app flow)
- `feature_scaler.pkl`: Feature scaler artifact (not auto-loaded in current app flow)

## Quick Start

1. Create/activate a virtual environment
2. Install dependencies
3. Launch Streamlit

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install streamlit numpy pandas plotly
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).



## Using the Real Model 


1. Load `best_aq_model.h5` with Keras
2. Load `feature_scaler.pkl` with joblib
3. Build the correct 48x15 feature window
4. Replace the `simulate_prediction(...)` call in `app.py` with actual model inference

> Note: The About section in the UI mentions `model.h5` and `scaler.pkl`; your current filenames are `best_aq_model.h5` and `feature_scaler.pkl`.

## Troubleshooting

- `ModuleNotFoundError`: Install missing package(s) in the active venv.
- Streamlit command not found: Ensure venv is activated before running `streamlit run app.py`.
- `ValueError: Seed must be between 0 and 2**32 - 1`: The simulator seed can become negative for some input combinations. Clamp/modulo the seed in `simulate_prediction(...)`.
- Port already in use: Run with a different port:

```zsh
streamlit run app.py --server.port 8502
```

## Future Improvements

- Connect OpenAQ/CPCB live feeds
- Add uncertainty estimates and forecast confidence intervals
- Add Docker + dependency manifest (`requirements.txt`)

