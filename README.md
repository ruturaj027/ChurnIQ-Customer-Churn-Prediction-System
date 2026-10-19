# ChurnIQ — Customer Churn Prediction

A polished customer churn dashboard built with **Flask**, **HTML/CSS**, and **scikit-learn**.

## UI features
- Responsive sidebar navigation and dashboard header
- Summary cards for the model, input features, and output
- Structured customer details form
- Prediction result panel with probability meter and risk category
- Input validation and friendly error messages
- Synthetic-data warning for responsible demo use

## Project structure
```text
ChurnIQ/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── churn_dataset.csv
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Run on Windows
Open this folder in VS Code, then run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
python app.py
```

Then open **http://127.0.0.1:5000**.

`train.py` creates `models/churn_pipeline.joblib`, which Flask loads for predictions.

> The included churn data is synthetic for learning and UI testing only. Do not present model scores as real-world performance.
