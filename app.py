
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

# Project paths
ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"

# Flask configuration
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)

DEFAULTS = {
    "tenure": "12",
    "monthly_charges": "75",
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "senior_citizen": "0",
}

OPTIONS = {
    "contract": ["Month-to-month", "One year", "Two year"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "tech_support": ["Yes", "No"],
    "paperless_billing": ["Yes", "No"],
    "payment_method": [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card",
    ],
    "senior_citizen": ["0", "1"],
}


@app.route("/", methods=["GET", "POST"])
def index():
    form = DEFAULTS.copy()
    result = None
    error = None

    if request.method == "POST":
        for key in form:
            form[key] = request.form.get(key, DEFAULTS[key])

        try:
            # Validate files
            if not (TEMPLATE_DIR / "index.html").is_file():
                raise FileNotFoundError(
                    f"Template missing: {TEMPLATE_DIR / 'index.html'}"
                )

            if not MODEL_PATH.is_file():
                raise FileNotFoundError(
                    "Trained model not found. Run "
                    "'python train.py' first."
                )

            # Validate numeric inputs
            tenure = int(form["tenure"])
            monthly_charges = float(form["monthly_charges"])
            senior_citizen = int(form["senior_citizen"])

            if not 0 <= tenure <= 72:
                raise ValueError(
                    "Tenure must be between 0 and 72 months."
                )

            if not 0 <= monthly_charges <= 200:
                raise ValueError(
                    "Monthly charges must be between 0 and 200."
                )

            if senior_citizen not in (0, 1):
                raise ValueError(
                    "Choose a valid senior-citizen value."
                )

            # Validate dropdown inputs
            for key, valid_options in OPTIONS.items():
                if form[key] not in valid_options:
                    raise ValueError(
                        f"Invalid value for {key.replace('_', ' ')}."
                    )

            # Prepare customer data
            customer = pd.DataFrame([{
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "contract": form["contract"],
                "internet_service": form["internet_service"],
                "tech_support": form["tech_support"],
                "paperless_billing": form["paperless_billing"],
                "payment_method": form["payment_method"],
                "senior_citizen": senior_citizen,
            }])

            # Load trained model
            model = joblib.load(MODEL_PATH)

            # Predict churn probability
            probability = float(
                model.predict_proba(customer)[0, 1]
            )

            prediction = int(probability >= 0.5)

            result = {
                "probability": round(probability * 100, 1),
                "prediction": (
                    "Likely to churn"
                    if prediction
                    else "Likely to stay"
                ),
                "risk": (
                    "High" if probability >= 0.7
                    else "Medium" if probability >= 0.4
                    else "Low"
                ),
                "is_churn": bool(prediction),
            }

        except (ValueError, TypeError, FileNotFoundError) as exc:
            error = str(exc)

        except Exception as exc:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Check your trained model "
                "and input feature names."
            )

    # Render dashboard
    return render_template(
        "index.html",
        form=form,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    print("Project folder:", ROOT)
    print("Template folder:", TEMPLATE_DIR)
    print("Template exists:", (TEMPLATE_DIR / "index.html").is_file())
    print("Model exists:", MODEL_PATH.is_file())

    app.run(debug=True)
