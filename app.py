import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Filter states
last_lat = None
last_lon = None
alpha = 0.35  # Smoothing factor for sensor jitter

@app.route('/')
def home():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error loading index.html: {str(e)}", 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "message": "Backend is running smoothly"}), 200

@app.route('/process-telemetry', methods=['POST', 'OPTIONS'])
def process_telemetry():
    global last_lat, last_lon

    # Pre-flight request handler for CORS
    if request.method == 'OPTIONS':
        return jsonify({"status": "preflight-ok"}), 200

    try:
        data = request.get_json(force=True, silent=True) or {}
        raw_lat = float(data.get('lat', 25.18))
        raw_lon = float(data.get('lon', 75.83))
        gyro = float(data.get('gyro', 0.05))
        gps_valid = bool(data.get('gps_valid', True))

        if last_lat is None:
            last_lat = raw_lat
            last_lon = raw_lon

        if gps_valid:
            # Low-pass filter for GPS smoothing
            filtered_lat = alpha * raw_lat + (1.0 - alpha) * last_lat
            filtered_lon = alpha * raw_lon + (1.0 - alpha) * last_lon
            last_lat = filtered_lat
            last_lon = filtered_lon

            # Drift estimation in meters
            dist_deg = np.sqrt((raw_lat - filtered_lat)**2 + (raw_lon - filtered_lon)**2)
            drift_error = round(float(dist_deg * 111000.0), 2)
            bias = round(float(np.sin(gyro) * 0.015), 4)
        else:
            # Dead reckoning prediction step
            step = 0.00007
            last_lat += step * np.cos(np.radians(45))
            last_lon += step * np.sin(np.radians(45))
            drift_error = round(float(np.random.uniform(1.5, 3.2)), 2)
            bias = round(float(0.038 + np.random.uniform(-0.004, 0.004)), 4)

        return jsonify({
            "status": "success",
            "backend_link": "connected",
            "ai_lat": round(last_lat, 6),
            "ai_lon": round(last_lon, 6),
            "drift_error_meters": drift_error,
            "predicted_bias": bias
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)