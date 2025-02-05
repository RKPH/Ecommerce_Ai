import pickle

def recommendation(model_path, p_id, top_k=100):
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    for product_id, recs in list(model.items()):
        if product_id == p_id:
            formatted_recommendations = {
                "product_id": product_id,
                "recommendations": [{"product_id": item} for item in recs]
            }

    return formatted_recommendations



