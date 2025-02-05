import pandas as pd
import pickle
import random
import os
import numpy as np

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


def get_user_rl_agent(all_product_ids, user_id, model_file):
    """
    Create or load an RL agent for the user.
    all_product_ids: list/array of all product_ids in the dataset.
    model_file: file path for saving/loading the model.
    """
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

        # 1) Find the most recent date the target product appeared
        target_max_date = df.loc[df["product_id"] == target_product_id, "event_time"].max()

        if pd.isna(target_max_date):
            # If the target item never appeared, fallback to normal CF approach or skip
            # For safety, let's just return an empty DataFrame or do a full fallback.
            print(f"Warning: target_product_id={target_product_id} never appeared in the data.")
            return pd.DataFrame(columns=["product_id", "predicted_score", "category_code", "brand", "appearance_count"])

        # 2) Look back 3 months from that item's last appearance date
        cutoff_date = target_max_date - pd.DateOffset(months=3)

        # 3) Filter the data between (cutoff_date) and (target_max_date)
        df_item_3_months = df[(df["event_time"] >= cutoff_date) & (df["event_time"] <= target_max_date)]

        # 4) Identify sessions where the target product appears in that time window
        filtered_sessions = df_item_3_months.loc[df_item_3_months["product_id"] == target_product_id, "user_session"].unique()
        related_sessions = df_item_3_months[df_item_3_months["user_session"].isin(filtered_sessions)]

        # 5) Identify candidate items (excluding the target_product_id)
        candidate_products = related_sessions["product_id"].unique()
        candidate_products = [p for p in candidate_products if p != target_product_id]

        # 6) Predict scores for each candidate item
        predicted_scores = []
        for p in candidate_products:
            est_score = cf_model.predict(target_user_id, p).est
            predicted_scores.append((p, est_score))

        # 7) Calculate the appearance counts for these items in the related sessions
        appearance_counts = related_sessions["product_id"].value_counts()

        predicted_scores_df = pd.DataFrame(predicted_scores, columns=["product_id", "predicted_score"])
        predicted_scores_df = predicted_scores_df.merge(product_details, on="product_id", how="left")

        # Attach the appearance_count column
        predicted_scores_df["appearance_count"] = predicted_scores_df["product_id"].map(appearance_counts)

        # 8) Prioritize items appearing >= 3 times
        high_frequency_items = predicted_scores_df[predicted_scores_df["appearance_count"] >= 3]
        high_frequency_items = high_frequency_items.sort_values(
            by=["appearance_count", "predicted_score"],
            ascending=[False, False]
        )
        top_high_freq = high_frequency_items.head(3)

        # 9) The remaining items, sorted by predicted_score
        remaining_items = predicted_scores_df[
            ~predicted_scores_df["product_id"].isin(top_high_freq["product_id"])]
        remaining_items = remaining_items.sort_values(by="predicted_score", ascending=False)

        # 10) Combine and take top_n
        recommendations = pd.concat([top_high_freq, remaining_items]).head(top_n)

        # 11) If we still don't have enough items to reach top_n
        if len(recommendations) < top_n:
            additional_needed = top_n - len(recommendations)
            leftover_products = [pid for pid in all_product_ids if pid not in recommendations["product_id"].values]

            additional_scores = []
            for pid in leftover_products:
                est_score = cf_model.predict(target_user_id, pid).est
                additional_scores.append((pid, est_score))

            additional_df = pd.DataFrame(additional_scores, columns=["product_id", "predicted_score"])
            additional_df = additional_df.merge(product_details, on="product_id", how="left")
            additional_df = additional_df.sort_values(by="predicted_score", ascending=False)

            # Take the top needed items
            add_top = additional_df.head(additional_needed)
            recommendations = pd.concat([recommendations, add_top])

        return recommendations.head(top_n)

    else:
        # --------------------------------------------------
        # (B) RL-based recommendation for cold-start (unknown user)
        # --------------------------------------------------
        print("User not found in model. Using RL-based recommendations for cold-start.")

        # 1) Create/Load the RL Agent
        user_model_file = get_user_model_file(target_user_id)
        agent = get_user_rl_agent(all_product_ids, target_user_id, user_model_file)

        # 2) Identify candidate items
        filtered_sessions = df[df["product_id"] == target_product_id]["user_session"].unique()
        related_sessions = df[df["user_session"].isin(filtered_sessions)]
        candidate_products = related_sessions["product_id"].unique()
        candidate_products = [p for p in candidate_products if p != target_product_id]

        # Determine fallback brand/category
        target_row = df[df["product_id"] == target_product_id].head(1)
        if not target_row.empty:
            t_brand = target_row["brand"].iloc[0]
            t_category = target_row["category_code"].iloc[0]
        else:
            t_brand = None
            t_category = None

        # If no candidate products
        if len(candidate_products) == 0:
            if t_brand is not None and focus_on_brand:
                fallback_candidates = df[df["brand"] == t_brand]["product_id"].unique()
            else:
                fallback_candidates = df[df["category_code"] == t_category]["product_id"].unique()

            if len(fallback_candidates) == 0:
                # fallback to all products
                fallback_candidates = all_product_ids

            # limit random sample
            max_sample = 50
            if len(fallback_candidates) > max_sample:
                fallback_candidates = random.sample(list(fallback_candidates), max_sample)

            candidate_products = fallback_candidates
        else:
            # If candidates exist but are fewer than top_n, expand
            if len(candidate_products) < top_n:
                candidate_df = df[df["product_id"].isin(candidate_products)]
                candidate_brands = candidate_df["brand"].dropna().unique()
                candidate_cats = candidate_df["category_code"].dropna().unique()

                if focus_on_brand and len(candidate_brands) > 0:
                    fallback_candidates = df[df["brand"].isin(candidate_brands)]["product_id"].unique()
                else:
                    fallback_candidates = df[df["category_code"].isin(candidate_cats)]["product_id"].unique()

                # limit random sample
                max_sample = 50
                if len(fallback_candidates) > max_sample:
                    fallback_candidates = random.sample(list(fallback_candidates), max_sample)

                # Combine
                candidate_products = list(set(candidate_products).union(set(fallback_candidates)))

        # 3) Prepare the state = (user_id, product_id, brand_idx, category_idx)
        brand_idx, cat_idx = get_brand_category_idx(brand_map, category_map, t_brand, t_category)
        state = np.array([target_user_id, target_product_id, brand_idx, cat_idx], dtype=np.float32)

        # 4) Agent uses Q-values to choose the top_n actions
        recommended_ids = agent.recommend(
            state=state,
            top_n=top_n,
            candidate_actions=candidate_products
        )

        # If it's still missing items for top_n, pick randomly
        if len(recommended_ids) < top_n:
            need_more = top_n - len(recommended_ids)
            fallback = [p for p in all_product_ids if p not in recommended_ids]
            if len(fallback) > need_more:
                fallback = random.sample(fallback, need_more)
            recommended_ids.extend(fallback)

        # 5) Determine the reward based on event_type
        reward_mapping = {"view": 1, "cart": 3, "purchase": 5}
        reward = reward_mapping.get(event_type, 1)

        # 6) Store experience for the first action & train
        if len(recommended_ids) > 0:
            first_action = recommended_ids[0]
        else:
            first_action = target_product_id

        agent.store_experience(state, first_action, reward, None)
        agent.train()
        agent.save_model()

        # Return DataFrame of recommendations
        rec_df = pd.DataFrame({"product_id": recommended_ids})
        rec_df = rec_df.merge(product_details, on="product_id", how="left")
        return rec_df.head(top_n)
