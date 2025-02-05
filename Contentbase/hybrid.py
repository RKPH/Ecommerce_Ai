import os.path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import time
import pickle
from Processdata.preprocessing import preprocess_data

# Function to preprocess and generate content-based similarity using ANN (Sklearn)
def content_base(df_content, k=5):
    # Kết hợp các thông tin cần thiết thành một văn bản duy nhất
    df_content['combined_features'] = (
            df_content['name'].astype(str) + " " +
            df_content['product_id'].astype(str) + " " +
            df_content['category_code'].astype(str) + " " +
            df_content['brand'].astype(str) + " " +
            df_content['price'].astype(str)
    )

    # Vectorize data
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
    tfidf_matrix = vectorizer.fit_transform(df_content['combined_features']).astype(np.float32)

    # Model
    model = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute', n_jobs=-1)
    model.fit(tfidf_matrix)

    # Get K nearest items
    distances, indices = model.kneighbors(tfidf_matrix)

    # Save result into dict
    recommendations = {}
    product_ids = df_content['product_id'].tolist()

    for i, p_id in enumerate(product_ids):
        recommendations[p_id] = [product_ids[idx] for idx in indices[i] if idx != i]  # remove it's self

    return recommendations

# # Main Function
# if __name__ == "__main__":
#     # Load data
#     bucket_name = "recommendation"
#     file_path = "merged_data_second.csv"
#     # Preprocess data và lấy phần dữ liệu mẫu
#     df, _ = preprocess_data(bucket_name=bucket_name, file_name=file_path, is_encoded=True)
#     print("Preprocess Complete !")
#
#     # Select features and Drop duplicate
#     selected_features = ["name", "product_id", "category_code", "brand", "price"]
#     df_content = df[selected_features].drop_duplicates(subset=['product_id'])
#     df_content = df_content.sort_values(by="product_id", ascending=False)
#     print("Drop duplicate and Feature Selection Complete !")
#
#     # Content Base
#     s_time = time.time()
#     k = 10  # Total recommended items
#     recommendations = content_base(df_content, k=k)
#     e_time = time.time()
#     print("Train Complete !")
#     print(f"Time consumed: {e_time - s_time}")
#
#     # Display result
#     for product_id, recs in list(recommendations.items())[:5]:
#         print(f"Product ID: {product_id} -> Recommended Products: {recs}")
#
#     if not os.path.exists("Model/"):
#         os.makedirs("Model/")
#
#     save_path = "Model/Test_content_base.pkl"
#     with open(save_path, 'wb') as f:
#         pickle.dump(recommendations, f)