# Stock Market Prediction App

This project predicts stock prices using a trained Keras model and displays the result in a Streamlit web app.

## Requirements

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Project files

- `app.py` – Streamlit app
- `model.weights.h5` – trained LSTM model weights
- `Stock_Market_Prediction_Model_Creation.ipynb` – notebook used to build the model

## Notes

- Make sure your environment has internet access for Yahoo Finance data.
- The app downloads stock data using `yfinance`.
