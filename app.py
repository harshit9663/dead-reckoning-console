from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__, static_folder='.')
CORS(app, resources={r"/*": {"origins": "*"}})

# Train lightweight drift model
model = RandomForestRegressor(n_estimators=25, random_state=42)
X_dummy = np.random.normal(0, 0.05, size=(100, 3))
Y_dummy = np.random.normal(0.045, 0.008, size=100)
model.fit(X_dummy, Y_dummy)

last_known_pos = None
current_heading_rad = 0.7854

# Direct root route to serve index.html
@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

# Health check route
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "dead-reckoning-engine"})

# Main telemetry processing endpoint
@app.route("/process-telemetry", methods=["POST", "OPTIONS"])
def process_telemetry():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    global last_known_pos, current_heading_rad
    data = request.get_json(force=True, silent=True) or {}
    
    lat = float(data.get("lat", 25.1800))
    lon = float(data.get("lon", 75.8300))
    raw_gyro = float(data.get("gyro", 0.02))
    gps_valid = bool(data.get("gps_valid", False))
    
    if last_known_pos is None:
        last_known_pos = [lat, lon]

    # Predict drift
    feats = np.array([[raw_gyro, raw_gyro * 0.94, 1.25]])
    predicted_bias = float(model.predict(feats)[0])
    corrected_gyro = raw_gyro - predicted_bias

    step_distance = 0.000075 

    if gps_valid:
        last_known_pos = [lat, lon]
        est_lat, est_lon = lat, lon
        drift_error = 0.0
    else:
        current_heading_rad += corrected_gyro * 0.15
        last_known_pos[0] += step_distance * np.cos(current_heading_rad)
        last_known_pos[1] += step_distance * np.sin(current_heading_rad)
        est_lat = last_known_pos[0]
        est_lon = last_known_pos[1]
        drift_error = float(np.sqrt((est_lat - lat)**2 + (est_lon - lon)**2) * 111000)

    return jsonify({
        "status": "success",
        "predicted_bias": round(predicted_bias, 4),
        "corrected_gyro": round(corrected_gyro, 4),
        "ai_lat": round(est_lat, 6),
        "ai_lon": round(est_lon, 6),
        "drift_error_meters": round(drift_error, 2)
    })

if __name__ == "__main__":
    print("\n[+] AI Dead Reckoning Engine Running on http://127.0.0.1:8080")
    app.run(host="127.0.0.1", port=8080, debug=False)