

\# AegisPulse

\### Predict Early. Protect Tomorrow.



AegisPulse is an AI-powered disaster risk monitoring and predictive analytics project focused on analyzing rainfall patterns across India. It uses historical rainfall data, statistical analysis, data processing, and geospatial visualization to explore rainfall trends and identify high-rainfall regions.



\## Project Overview



The project analyzes daily rainfall data for India during 2024 to understand rainfall distribution, seasonal patterns, and geographical variations.



\## Features



\- Rainfall dataset inspection and preprocessing

\- Missing value detection and data cleaning

\- Annual and monthly rainfall analysis

\- Rainfall distribution visualization

\- Identification of high-rainfall locations

\- Interactive India rainfall maps

\- Rainfall classification and feature preparation



\## Technologies Used



\- Python

\- NumPy

\- Pandas

\- Xarray

\- SciPy

\- Matplotlib

\- Seaborn

\- Folium

\- NetCDF



\## Project Structure



AegisPulse/

├── data/

│   └── processed/

├── src/

│   ├── analyze\_rainfall.py

│   ├── clean\_data.py

│   ├── feature\_engineering.py

│   ├── rainfall\_anomaly.py

│   ├── rainfall\_map.py

│   └── rainfall\_risk\_map.py

├── .gitignore

├── requirements.txt

└── README.md



\## Installation



1\. Clone the repository:



&#x20;  git clone YOUR\_GITHUB\_REPOSITORY\_URL



2\. Navigate to the project directory:



&#x20;  cd AegisPulse



3\. Create a virtual environment:



&#x20;  python -m venv venv



4\. Activate the environment on Windows:



&#x20;  venv\\Scripts\\activate



5\. Install dependencies:



&#x20;  pip install -r requirements.txt



\## Dataset



The project uses historical Indian rainfall data in NetCDF format for 2024.



The original dataset is not included in this repository. Place the downloaded dataset inside the data/raw/ directory before running the preprocessing scripts.



\## Current Project Status



Version 1.0 – Initial rainfall analysis and geospatial visualization prototype.



Future development will focus on machine learning-based rainfall forecasting, flood-risk prediction, and advanced disaster monitoring.



\## Author



Developed as a Data Science and Predictive Analytics project.



