# -----------------------------------------------------------------------------------------------------------------------
# IMPORTS & MODULES
# -----------------------------------------------------------------------------------------------------------------------

from pathlib import Path

from flask import Flask, request, jsonify, render_template

from ..classifier import TitanicClassifier

# -----------------------------------------------------------------------------------------------------------------------
# FLASK APPLICATION: Main Application Endpoint
# -----------------------------------------------------------------------------------------------------------------------

def create_app(filepath: Path) -> Flask:

    app = Flask(__name__)

    classifier = TitanicClassifier(data_path=filepath)

    @app.post("/predict/log")
    def predict_cls():

        payload = request.get_json(silent=True)      

        if not isinstance(payload, dict):
            return jsonify({
                "error": "Expected a JSON object."
            }), 400

        try:
            prediction = classifier.predict_cls(passenger=payload)
            if prediction is None:
                return jsonify("No prediction produced.")

            return jsonify(prediction)

        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/predict/mlp")
    def predict_mlp():
        
        payload = request.get_json(silent=True)
        
        if not isinstance(payload, dict):
            return jsonify({
                "error": "Expected a JSON object."
            }), 400
        
        try:
            prediction = classifier.predict_mlp(passenger=payload)
            if prediction is None:
                return jsonify("No prediction produced.")
        
            return jsonify(prediction)
        
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/metrics/log")
    def metrics_cls():

        return render_template(
            "metrics.html", 
            metrics=classifier.diagnostics_cls
        )

    @app.get("/metrics/mlp")
    def metrics_mlp():
    
        return render_template(
            "metrics.html", 
            metrics=classifier.diagnostics_mlp
        )

    @app.get("/matrix/log")
    def matrix_cls():
        return render_template(
            "matrix.html",
            matrix=classifier.cls_matrix,
            labels=[0, 1],
        )

    @app.get("/matrix/mlp")
    def matrix_mlp():
        return render_template(
            "matrix.html",
            matrix=classifier.mlp_matrix,
            labels=[0, 1],
        )

    @app.get("/")
    def home():
        return render_template("index.html")

    return app