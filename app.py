from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__, template_folder="templates")

# Load models
binary_scaler = joblib.load("binary_scaler.pkl")
binary_model = joblib.load("xgboost_multiclass_model.pkl")
multiclass_model = joblib.load("stacking_multiclass_model.pkl")

binary_features = [
    'Age_day', 'last_funding_at', 'milestones', 'funding_total_usd',
    'first_funding_at', 'first_milestone_at', 'last_milestone_at',
    'relationships', 'funding_rounds', 'lng', 'lat', 'funding_per_round'
]

if hasattr(multiclass_model, "feature_names_in_"):
    multiclass_features = multiclass_model.feature_names_in_
else:
    multiclass_features = []


def prepare_features(data):
    Age_day = (2025 - data['founded_at']) * 365
    funding_per_round = data['funding_total_usd'] / max(data['funding_rounds'], 1)

    features = {
        'founded_at': data['founded_at'],
        'first_funding_at': data['first_funding_at'],
        'last_funding_at': data['last_funding_at'],
        'funding_rounds': data['funding_rounds'],
        'funding_total_usd': data['funding_total_usd'],
        'first_milestone_at': data['first_milestone_at'],
        'last_milestone_at': data['last_milestone_at'],
        'milestones': data['milestones'],
        'relationships': data['relationships'],
        'investment_rounds': data['investment_rounds'],
        'lat': data['lat'],
        'lng': data['lng'],
        'Age_day': Age_day,
        'funding_per_round': funding_per_round,
        f'category_{data["category"]}': 1,
        f'country_{data["country"]}': 1
    }

    return pd.DataFrame([features])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict-binary", methods=["POST"])
def predict_binary():
    try:
        data = request.get_json()
        input_df = prepare_features(data)
        X_binary = input_df[binary_features]
        X_scaled = binary_scaler.transform(X_binary)
        binary_pred = binary_model.predict(X_scaled)
        status = "Active" if binary_pred[0] == 1 else "Closed"
        return jsonify({"binary_prediction": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict-multiclass", methods=["POST"])
def predict_multiclass():
    try:
        data = request.get_json()
        input_df = prepare_features(data)

        df_dummies = pd.get_dummies(input_df)
        for col in multiclass_features:
            if col not in df_dummies.columns:
                df_dummies[col] = 0
        df_dummies = df_dummies[multiclass_features]

        prediction = multiclass_model.predict(df_dummies)[0]
        status_map = {
            0: "Acquired",
            1: "Operating",
            2: "IPO",
            3: "Closed"
        }
        return jsonify({
            "multiclass_prediction": int(prediction),
            "status": status_map.get(int(prediction), "Unknown")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
