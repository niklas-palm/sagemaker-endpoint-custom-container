import os
import logging

from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

from src.inference import load_model, predict

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model from the SM_MODEL_DIR environment variable
model_path = os.environ.get("SM_MODEL_DIR")
if not model_path:
    raise ValueError("SM_MODEL_DIR environment variable is not set.")
model = load_model(model_path)

# Use ProxyFix middleware for running behind a proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)


@app.route("/ping", methods=["GET"])
def ping():
    """
    Healthcheck function.
    """
    return "pong"


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Endpoint for model invocations.
    """
    try:
        body = request.json
        if not body:
            return jsonify({"error": "No JSON data provided"}), 400

        prediction = predict(body, model)
        return jsonify(prediction)
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": "An error occurred during prediction"}), 500


if __name__ == "__main__":
    app.run(debug=True)
