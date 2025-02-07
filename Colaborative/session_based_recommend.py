import pandas as pd
import pickle
import random
import os
import numpy as np
import logging
from .rl_agent import DQNAgent

# -----------------------------------------------------------
# 1) Create brand_map and category_map (done once for the entire system)
# -----------------------------------------------------------
def build_mappings(df):
    """
    Create dictionaries to map brand -> brand_idx, category -> category_idx.
    (0 is "unknown")
    """
    unique_brands = df['brand'].dropna().unique()
    brand_map = {b: i+1 for i, b in enumerate(unique_brands)}  # start index=1
    brand_map["unknown"] = 0

    unique_categories = df['category_code'].dropna().unique()
    category_map = {c: i+1 for i, c in enumerate(unique_categories)}
    category_map["unknown"] = 0

    return brand_map, category_map


def get_brand_category_idx(brand_map, category_map, brand, category):
    """
    Get brand_idx, category_idx from the corresponding maps.
    """
    brand_idx = brand_map.get(brand, 0)       # 0 if not found
    category_idx = category_map.get(category, 0)
    return brand_idx, category_idx


# -----------------------------------------------------------
# 2) RL Agent for each user (can be loaded or newly created)
# -----------------------------------------------------------
def get_user_model_file(user_id):
    return f"dqn_model_{user_id}.h5"


def check_user_in_model(cf_model, user_id):
    """
    Check if the user_id exists in the collaborative filtering model.
    """
    try:
        cf_model.trainset.to_inner_uid(user_id)
        return True
    except ValueError:
        return False


def get_user_rl_agent(df, user_id):
    """Load or initialize an RL agent for a specific user."""
    model_file = get_user_model_file(user_id)

    # Initialize the agent with all product IDs
    all_product_ids = df["product_id"].unique()
    agent = DQNAgent(action_space=all_product_ids, model_file=model_file)

    if os.path.exists(model_file):
        print(f"Loading RL model for user {user_id}...")
        agent.load_model()
    else:
        print(f"Creating new RL model for user {user_id}...")

    return agent


# -----------------------------------------------------------
# 3) Main recommendation function
# -----------------------------------------------------------
def recommend_products(
        cf_model,  # Now pass cf_model directly, no need for model_file
        df,
        target_user_id,
        target_product_id,
        event_type,
        top_n=10,
        focus_on_brand=False
):
    """
    If the user exists in the CF model -> use collaborative filtering
    Otherwise -> use RL-based approach for cold-start
    """
    # Build brand_map and category_map (or load them if you already have them saved)
    brand_map, category_map = build_mappings(df)

    # Ensure df['event_time'] is datetime
    df["event_time"] = pd.to_datetime(df["event_time"])

    # Check if the user exists in the CF model
    user_exists = check_user_in_model(cf_model, target_user_id)

    # Common data
    product_details = df[["product_id", "category_code", "brand"]].drop_duplicates(subset=["product_id"])
    all_product_ids = df["product_id"].unique()

    if user_exists:
        # --------------------------------------------------
        # (A) CF-based recommendation for known users
        # --------------------------------------------------
        target_max_date = df.loc[df["product_id"] == target_product_id, "event_time"].max()

        if pd.isna(target_max_date):
            logging.info(f"Warning: target_product_id={target_product_id} never appeared in the data.")
            return pd.DataFrame(columns=["product_id", "predicted_score", "category_code", "brand", "appearance_count"])

        cutoff_date = target_max_date - pd.DateOffset(months=3)
        df_item_3_months = df[(df["event_time"] >= cutoff_date) & (df["event_time"] <= target_max_date)]
        filtered_sessions = df_item_3_months.loc[df_item_3_months["product_id"] == target_product_id, "user_session"].unique()
        related_sessions = df_item_3_months[df_item_3_months["user_session"].isin(filtered_sessions)]
        candidate_products = related_sessions["product_id"].unique()
        candidate_products = [p for p in candidate_products if p != target_product_id]

        predicted_scores = [(p, cf_model.predict(target_user_id, p).est) for p in candidate_products]
        appearance_counts = related_sessions["product_id"].value_counts()
        predicted_scores_df = pd.DataFrame(predicted_scores, columns=["product_id", "predicted_score"])
        predicted_scores_df = predicted_scores_df.merge(product_details, on="product_id", how="left")
        predicted_scores_df["appearance_count"] = predicted_scores_df["product_id"].map(appearance_counts)

        high_frequency_items = predicted_scores_df[predicted_scores_df["appearance_count"] >= 3]
        high_frequency_items = high_frequency_items.sort_values(by=["appearance_count", "predicted_score"], ascending=[False, False])
        top_high_freq = high_frequency_items.head(3)

        remaining_items = predicted_scores_df[~predicted_scores_df["product_id"].isin(top_high_freq["product_id"])]
        remaining_items = remaining_items.sort_values(by="predicted_score", ascending=False)

        recommendations = pd.concat([top_high_freq, remaining_items]).head(top_n)

        return recommendations.head(top_n)

    else:
        # -----------------------------
        # RL-based Cold-Start Approach
        # -----------------------------
        logging.info("User not found in model. Using RL-based recommendations for cold-start.")

        agent = get_user_rl_agent(df, target_user_id)

        # Generate candidate items
        filtered_sessions = df[df["product_id"] == target_product_id]["user_session"].unique()
        related_sessions = df[df["user_session"].isin(filtered_sessions)]
        candidate_products = related_sessions["product_id"].unique()
        candidate_products = [p for p in candidate_products if p != target_product_id]

        # 3) If no candidate, fallback to random
        if len(candidate_products) == 0:
            all_products = df["product_id"].unique()
            if len(all_products) > top_n:
                recommended_ids = random.sample(list(all_products), top_n)
            else:
                recommended_ids = list(all_products)
            print("No candidate items => random fallback.")
        else:
            # RL-based ranking
            state = (target_user_id, target_product_id)
            recommended_ids = agent.recommend(
                state=state,
                top_n=top_n,
                candidate_actions=candidate_products
            )

        # Convert event type to reward
        reward_mapping = {"view": 1, "cart": 3, "purchase": 5}
        reward = reward_mapping.get(event_type, 1)

        # Store and train on user-specific experience
        if len(recommended_ids) < top_n:
            remaining_needed = top_n - len(recommended_ids)
            all_products = df["product_id"].unique()
            fallback_items = [p for p in all_products if p not in recommended_ids]
            fallback_items = random.sample(fallback_items, min(len(fallback_items), remaining_needed))
            recommended_ids.extend(fallback_items)

        if len(recommended_ids) > 0:
            first_action = recommended_ids[0]
        else:
            first_action = target_product_id

        agent.store_experience(state, first_action, reward, None)
        agent.train()

        # Save the updated agent for the user
        agent.save_model()
        # Build a DataFrame for recommendations
        product_details = df[["product_id", "category_code", "brand"]].drop_duplicates(subset=["product_id"])
        recommendations_df = pd.DataFrame({"product_id": recommended_ids})
        recommendations_df = recommendations_df.merge(product_details, on="product_id", how="left")

        return recommendations_df
