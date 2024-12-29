from flask import Flask, request, jsonify
from flask_cors import CORS
from hybrid import Combined
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
logging.basicConfig(level=logging.DEBUG)

MODEL_FILE = "model.pkl"

@app.route('/predict', methods=['POST'])
def predict_api():
    logging.info("test");
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        product_id = float(data.get('product_id'))
        content_w = 0.7
        item_w = 0.3
        k_out = 10
        logging.info(type(product_id))
        logging.info(f"Received request with product_id: {product_id}")

        # Kiểm tra đầu vào hợp lệ
        if product_id is None:
            logging.error("Missing product_id")
            return jsonify({"error": "Missing product_id"}), 400

        if not (0 <= content_w <= 1 and 0 <= item_w <= 1):
            logging.error("Invalid weight values: content_w=%s, item_w=%s", content_w, item_w)
            return jsonify({"error": "Weights must be between 0 and 1"}), 400

        # Gọi mô hình để dự đoán
        recommendations = Combined(product_id, content_w, item_w, k_out)

        # Log the recommendations for debugging
        logging.info(f"Recommendations: {recommendations}")

        return jsonify({"product_id": product_id, "recommendations": recommendations}), 200

    except Exception as e:
        # Log the complete exception for debugging
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
