from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from Contentbase.hybrid_recommendation import recommendation
from Colaborative.session_based_recommend import recommend_products
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np  # For appending to LabelEncoder classes
from prometheus_flask_exporter import PrometheusMetrics
app = Flask(__name__)
metrics = PrometheusMetrics(app, group_by='endpoint', default_latency_as_histogram=False)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
logging.basicConfig(level=logging.DEBUG)

# Load the dataset and preprocess
csv_file_path = "Data/dataset.csv"
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"Dataset file not found at {csv_file_path}")

df = pd.read_csv(csv_file_path)
enc_product_id = LabelEncoder()
enc_user_id = LabelEncoder()
enc_user_session = LabelEncoder()

# Fit and transform each column independently
df["product_id"] = enc_product_id.fit_transform(df["product_id"])
df["user_id"] = enc_user_id.fit_transform(df["user_id"])
df["user_session"] = enc_user_session.fit_transform(df["user_session"])
model_file = 'Model/model.pkl'
model_path = 'Model/content_base.pkl'
def update_label_encoder(encoder, value):
    """Dynamically update LabelEncoder with new values."""
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]

def update_label_encoder_if_exists(encoder, value):
    """Check if a value exists in the LabelEncoder. If it exists, encode it. Otherwise, return None."""
    if value in encoder.classes_:
        return encoder.transform([value])[0]  # Encode the value
    return None  # Return None if the value is not in the encoder

@app.route('/metrics')
def metrics_endpoint():
    return metrics.do_export()

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        # Retrieve and validate the product_id from the request
        data = request.get_json()
        product_id = data.get('product_id')

        if product_id is None:
            return jsonify({"error": "Missing product_id"}), 400

        try:
            product_id = int(product_id)
        except ValueError:
            return jsonify({"error": "product_id must be an integer"}), 400

        # Dynamically update the LabelEncoder for unseen product IDs
        try:
            encoded_product_id = update_label_encoder(enc_product_id, product_id)
            logging.info(f"Encoded product_id: {encoded_product_id}")
        except Exception as e:
            logging.error(f"Error during encoding: {str(e)}")
            return jsonify({"error": "Invalid product_id after encoding"}), 400

        # Call the recommendation logic with encoded product ID
        recommendation_result = recommendation(model_path, encoded_product_id)

        # Log raw recommendations
        logging.info(f"Raw recommendations before decoding: {recommendation_result}")

        # Validate the response format
        if not isinstance(recommendation_result, dict) or "recommendations" not in recommendation_result:
            return jsonify({"error": "Invalid response format from recommendation function"}), 500

        # Extract recommended product IDs
        recommended_product_ids = [item["product_id"] for item in recommendation_result["recommendations"]]

        # Decode the product IDs
        decoded_recommendations = enc_product_id.inverse_transform(np.array(recommended_product_ids))
        logging.info(f"Decoded recommendations: {decoded_recommendations}")

        # Return recommendations in JSON format
        return jsonify({
            'product_id': product_id,
            'recommendations': [{"product_id": int(prod_id)} for prod_id in decoded_recommendations]
        }), 200

    except Exception as e:
        logging.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route('/session-recommend', methods=['POST'])
def session_recommend_api():
    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")
        
        # Extract user and product IDs
        user_id = data.get('user_id')
        product_id = data.get('product_id')
        event_type = data.get('event_type')

        if user_id is None or product_id is None:
            return jsonify({"error": "Missing user_id or product_id"}), 400

        try:
            user_id = int(user_id)
            product_id = int(product_id)
        except ValueError:
            return jsonify({"error": "user_id and product_id must be integers"}), 400

        if user_id in enc_user_id.classes_:
            encoded_user_id = update_label_encoder(enc_user_id, user_id)
            logging.info(f"Encoded user_id: {encoded_user_id}")
        else:
            # Use the user_id directly if it doesn't exist in the encoder
            encoded_user_id = user_id
            logging.info(f"User ID not in encoder, using directly: {encoded_user_id}")
        # Dynamically update the LabelEncoder for unseen IDs
        try:
            encoded_product_id = update_label_encoder(enc_product_id, product_id)
            logging.info(f"user_id: {encoded_user_id}, product_id: {encoded_product_id} , event_type: {event_type}")
        except Exception as e:
            logging.error(f"Error during encoding: {str(e)}")
            return jsonify({"error": "Invalid user_id or product_id after encoding"}), 400

        # Call the session-based recommendation logic with encoded IDs
        recommendations_df = recommend_products(model_file, df, encoded_user_id, encoded_product_id,event_type)
        logging.info(f"Recommendations DataFrame columns: {recommendations_df.columns}")

        # Decode the product IDs in the recommendations
        recommendations_df['product_id'] = recommendations_df['product_id'].astype(int)
        recommendations_df['product_id'] = enc_product_id.inverse_transform(recommendations_df['product_id'])

        # Convert DataFrame to JSON-serializable structure
        if isinstance(recommendations_df, pd.DataFrame):
            recommendations = recommendations_df.to_dict(orient="records")  # Convert to list of dictionaries
        else:
            recommendations = recommendations_df  # Assuming it's already JSON-serializable

        # Log and return the recommendations    
        logging.info(f"Session-based Recommendations: {recommendations}")
        return jsonify({
            "user_id": user_id,  # Return original user_id
            "product_id": product_id,  # Return original product_id
            "recommendations": recommendations
        }), 200

    except Exception as e:
        logging.error(f"Error in session-recommend endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)