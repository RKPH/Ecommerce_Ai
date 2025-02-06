from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from Contentbase.hybrid_recommendation import recommendation
from Colaborative.session_based_recommend import recommend_products
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from prometheus_flask_exporter import PrometheusMetrics
import pickle
from io import BytesIO
from minio import Minio

app = Flask(__name__)
metrics = PrometheusMetrics(app, group_by='endpoint', default_latency_as_histogram=False)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.DEBUG)

# =========================
# MinIO Configuration
# =========================
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', '103.155.161.94:9000')  # Replace with your endpoint
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_SECURE = False  # Set to True if your MinIO uses HTTPS

BUCKET_NAME = 'models'             # Replace with your bucket name
SESSION_MODEL_NAME = 'model.pkl'   # Path to the session model in your bucket
CONTENT_MODEL_NAME = 'content_base.pkl'  # Path to the content-based model in your bucket

# Initialize the MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

def load_model_from_minio(bucket_name, object_name):
    """
    Retrieve and load a pickle model from a MinIO bucket.
    """
    try:
        # Get the object from MinIO
        response = minio_client.get_object(bucket_name, object_name)
        # Read the entire object data into memory
        model_data = response.read()
        response.close()
        response.release_conn()
        # Load and return the model from the data stream
        model = pickle.load(BytesIO(model_data))
        logging.info(f"Loaded model '{object_name}' from MinIO successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model '{object_name}' from MinIO: {str(e)}")
        raise

# Load models from MinIO at startup
try:
    session_model = load_model_from_minio(BUCKET_NAME, SESSION_MODEL_NAME)
    content_model = load_model_from_minio(BUCKET_NAME, CONTENT_MODEL_NAME)
except Exception as e:
    logging.error("Failed to load models from MinIO. Exiting application.")
    raise

# =========================
# Load Dataset from MinIO
# =========================
# =========================
# Load Dataset from MinIO
# =========================
def load_dataset_from_minio(bucket_name, object_name):
    """
    Retrieve and load a CSV dataset from a MinIO bucket.
    """
    try:
        # Get the object from MinIO
        response = minio_client.get_object(bucket_name, object_name)
        # Read the entire object data into memory
        dataset_data = response.read()
        response.close()
        response.release_conn()
        # Convert the byte data into a pandas DataFrame
        dataset_df = pd.read_csv(BytesIO(dataset_data))
        logging.info(f"Loaded dataset '{object_name}' from MinIO successfully.")
        return dataset_df
    except Exception as e:
        logging.error(f"Error loading dataset '{object_name}' from MinIO: {str(e)}")
        raise

# Specify the correct bucket name for the dataset
dataset_bucket_name = 'datasets'  # Correct bucket name
csv_file_path = "dataset.csv"  # Path to the dataset in the 'datasets' bucket

# Load dataset from MinIO
df = load_dataset_from_minio(dataset_bucket_name, csv_file_path)

# =========================
# LabelEncoders Setup
# =========================
enc_product_id = LabelEncoder()
enc_user_id = LabelEncoder()
enc_user_session = LabelEncoder()

# Fit and transform each column independently
df["product_id"] = enc_product_id.fit_transform(df["product_id"])
df["user_id"] = enc_user_id.fit_transform(df["user_id"])
df["user_session"] = enc_user_session.fit_transform(df["user_session"])

def update_label_encoder(encoder, value):
    """Dynamically update LabelEncoder with new values."""
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]

def update_label_encoder_if_exists(encoder, value):
    """Encode a value only if it exists in the LabelEncoder; otherwise return None."""
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return None

# =========================
# Endpoints
# =========================
@app.route('/metrics')
def metrics_endpoint():
    return metrics.do_export()

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        if product_id is None:
            return jsonify({"error": "Missing product_id"}), 400

        try:
            product_id = int(product_id)
        except ValueError:
            return jsonify({"error": "product_id must be an integer"}), 400

        # Dynamically update the product_id encoder if necessary
        try:
            encoded_product_id = update_label_encoder(enc_product_id, product_id)
            logging.info(f"Encoded product_id: {encoded_product_id}")
        except Exception as e:
            logging.error(f"Error during encoding: {str(e)}")
            return jsonify({"error": "Invalid product_id after encoding"}), 400

        # Call the recommendation function using the content-based model loaded from MinIO.
        # (Ensure your recommendation() function is updated to accept a model object.)
        recommendation_result = recommendation(content_model, encoded_product_id)
        logging.info(f"Raw recommendations before decoding: {recommendation_result}")

        if not isinstance(recommendation_result, dict) or "recommendations" not in recommendation_result:
            return jsonify({"error": "Invalid response format from recommendation function"}), 500

        # Extract recommended product IDs and decode them
        recommended_product_ids = [item["product_id"] for item in recommendation_result["recommendations"]]
        decoded_recommendations = enc_product_id.inverse_transform(np.array(recommended_product_ids))
        logging.info(f"Decoded recommendations: {decoded_recommendations}")

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

        # Extract user and product IDs along with the event type
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

        # Encode user_id: if not already in the encoder, update it dynamically
        if user_id in enc_user_id.classes_:
            encoded_user_id = update_label_encoder(enc_user_id, user_id)
            logging.info(f"Encoded user_id: {encoded_user_id}")
        else:
            encoded_user_id = user_id
            logging.info(f"User ID not in encoder, using directly: {encoded_user_id}")

        # Encode product_id dynamically
        try:
            encoded_product_id = update_label_encoder(enc_product_id, product_id)
            logging.info(f"user_id: {encoded_user_id}, product_id: {encoded_product_id}, event_type: {event_type}")
        except Exception as e:
            logging.error(f"Error during encoding: {str(e)}")
            return jsonify({"error": "Invalid user_id or product_id after encoding"}), 400

        # Call the session-based recommendation function using the session model loaded from MinIO.
        # (Ensure your recommend_products() function is updated to accept a model object.)
        recommendations_df = recommend_products(session_model, df, encoded_user_id, encoded_product_id, event_type)
        logging.info(f"Recommendations DataFrame columns: {recommendations_df.columns}")

        # Decode the product IDs in the recommendations
        recommendations_df['product_id'] = recommendations_df['product_id'].astype(int)
        recommendations_df['product_id'] = enc_product_id.inverse_transform(recommendations_df['product_id'])

        # Convert the DataFrame to a JSON-serializable list of dictionaries
        if isinstance(recommendations_df, pd.DataFrame):
            recommendations = recommendations_df.to_dict(orient="records")
        else:
            recommendations = recommendations_df

        logging.info(f"Session-based Recommendations: {recommendations}")
        return jsonify({
            "user_id": user_id,
            "product_id": product_id,
            "recommendations": recommendations
        }), 200

    except Exception as e:
        logging.error(f"Error in session-recommend endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
