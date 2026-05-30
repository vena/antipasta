"""
ML Server Factory
-----------------
Provides the public API for the AI backend. Starts immediately to allow 
the Logic controller to monitor initialization progress via the /info/ endpoint.
"""

import os
import logging
from flask import Flask, request, jsonify
from waitress import serve

logger = logging.getLogger("AntiPasta.MLServer")
EXPECTED_TOKEN = os.environ.get('ML_API_TOKEN', 'internal_secret_token')

def start_ml_server(engine, port=3333):
    """
    Initializes and starts the Waitress WSGI server.
    
    Args:
        engine: An instance of AntiPastaEngine (ml/engine.py).
    """
    app = Flask(__name__)

    @app.route('/p/', methods=['POST'])
    def predict():
        """Main inference endpoint. Returns 503 if the model is still loading."""
        if request.headers.get('Authorization', '') != f'Bearer {EXPECTED_TOKEN}':
            return jsonify({"error": "Unauthorized"}), 401
            
        if not engine.ready:
            return jsonify({
                "error": "AI Engine is still initializing",
                "status": engine.status
            }), 503
            
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided in POST body"}), 400
            
        try:
            img_bytes = request.files['image'].read()
            detections = engine.infer(img_bytes)
            return jsonify({"detections": detections})
            
        except ValueError as ve:
            logger.warning(f"Invalid payload for inference: {ve}")
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route('/info/', methods=['GET'])
    def info():
        """
        Diagnostic endpoint. Returns the current readiness state and 
        hardware acceleration metadata.
        """
        return jsonify({
            "ready": engine.ready,
            "status": engine.status,
            "engine": os.environ.get('ML_ENGINE', 'yolov11'),
            "hardware": {
                "provider": engine.active_provider,
                "device": engine.active_device
            }
        })

    @app.route('/hc/', methods=['GET'])
    def health():
        return "OK", 200

    logger.info(f"Starting ML API Server on port {port}...")
    
    # Note on threads=1: Concurrent calls to the same session
    # from different threads could cause segmentation faults.
    serve(app, host='0.0.0.0', port=port, threads=1)