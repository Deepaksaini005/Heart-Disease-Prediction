# Heart Disease Prediction (Streamlit)

A simple Streamlit app that predicts heart disease risk using a trained KNN model.

## Files

- `app.py` — Streamlit application
- `KNN_heart.pkl` — trained KNN model
- `scaler.pkl` — preprocessing scaler
- `columns.pkl` — ordered feature columns used by the model
- `heart.csv` — dataset
- `heart.ipynb` — notebook reference

## Setup

Install the required packages:

```powershell
cd "d:\Machine Learning\Heart Deases project"
python -m pip install -r requirements.txt
```

## Run

```powershell
cd "d:\Machine Learning\Heart Deases project"
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal (usually `http://localhost:8501`).

## Notes

- The app loads the model and preprocessing files from the project root.
- This is a Streamlit-only project; Django and MongoDB are not required.
