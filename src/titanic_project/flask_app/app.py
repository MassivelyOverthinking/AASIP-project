# -----------------------------------------------------------------------------------------------------------------------
# IMPORTS & MODULES
# -----------------------------------------------------------------------------------------------------------------------

from pathlib import Path

from flask import Flask, request, jsonify

from ..classifier import TitanicClassifier

# -----------------------------------------------------------------------------------------------------------------------
# FLASK APPLICATION: Main Application Endpoint
# -----------------------------------------------------------------------------------------------------------------------

def create_app(filepath: Path) -> Flask:

    app = Flask(__name__)

    classifier = TitanicClassifier(data_path=filepath)

    @app.post("/predict")
    def predict():

        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "error": "Expected a JSON object."
            }), 400

        passenger = payload.get("passenger")

        try:
            predictions = classifier.predict(passenger)
            if "passengers" in payload:
                return jsonify({"predictions": predictions})

            return jsonify(predictions[0])

        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/metrics")
    def metrics():
        return jsonify(classifier.diagnostics)

    return app