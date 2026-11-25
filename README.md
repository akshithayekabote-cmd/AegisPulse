# AEGISPULSE
### Predict Early. Protect Tomorrow.

An AI-powered rainfall prediction and regional risk analysis platform designed to identify potential rainfall events across India using historical rainfall patterns and spatial weather features.

## Overview

AegisPulse is a data science and predictive analytics project that uses historical daily rainfall data from the India Meteorological Department (IMD) to predict the probability of rainfall at individual geographical grid locations.

The platform combines machine learning, spatial analysis, historical rainfall patterns, and interactive visualization to make rainfall predictions easier to explore and understand.

## Key Features

- **Next-Day Rainfall Prediction:** Estimates the probability of rainfall at geographical grid locations.
- **Spatial Analysis:** Incorporates neighbouring grid locations to capture regional rainfall patterns.
- **Risk Classification:** Groups prediction probabilities into Low, Moderate, High, and Very High categories.
- **Interactive India Map:** Visualizes geographical rainfall predictions.
- **Prediction Analytics:** Displays rainfall probability distributions and key prediction statistics.
- **Regional Analysis:** Provides regional insights across mapped meteorological subdivisions.
- **Streamlit Dashboard:** Presents prediction results through an interactive dashboard.

## Technology Stack

- Python
- Pandas and NumPy
- Xarray and SciPy
- Streamlit
- PyDeck
- Folium
- NetCDF

## Dataset

The project uses daily gridded rainfall data from the India Meteorological Department (IMD).

- Data period: 2015–2024
- Total historical days: 3,653
- Geographical grid locations: 4,885
- Spatial resolution: 0.25°

The dataset was cleaned and processed to prepare historical rainfall features for model development.

## Prediction Model

AegisPulse uses a custom logistic regression-based prediction model with 15 input features.

The model incorporates:

- Historical rainfall values
- Previous-day and multi-day rainfall lags
- Rolling rainfall statistics
- Seasonal and calendar features
- Neighbouring-location rainfall statistics

Model V3 achieved the following results on the 2024 chronological test dataset:

| Evaluation Metric | Score |
|---|---:|
| F1 Score | 0.7071 |
| Precision | 0.6485 |
| Recall | 0.7773 |
| Accuracy | 0.8137 |

These results represent historical test performance and do not guarantee future prediction accuracy.

## Spatial Coverage

The regional subdivision mapping covers:

- 4,562 mapped grid locations
- 323 unmapped grid locations
- 34 mapped meteorological subdivisions
- Approximately 93.39% grid-location mapping coverage

## Project Structure

```text
AegisPulse/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── features/
│       ├── features_v3/
│       ├── predictions/
│       └── spatial_features/
│
├── models/
│   └── historical_rainfall_model_v3.npz
│
└── src/
    └── predict_next_day.py
```

## Installation and Setup

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project directory:

```bash
cd AegisPulse
```

3. Create a virtual environment:

```bash
py -m venv venv
```

4. Activate the virtual environment on Windows:

```powershell
venv\Scripts\Activate.ps1
```

5. Install dependencies:

```bash
py -m pip install -r requirements.txt
```

6. Run the Streamlit dashboard:

```bash
py -m streamlit run app.py
```

## Important Limitations and Disclaimer

- The current prediction pipeline uses historical rainfall data