from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pickle

def train_model(df_weighted, model_file="Model/model.pkl"):
    # Define rating scale
    min_score = df_weighted["score"].min()
    max_score = df_weighted["score"].max()
    reader = Reader(rating_scale=(min_score, max_score))

    # Load data for Surprise
    data = Dataset.load_from_df(df_weighted, reader)

    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.2)

    # Train the model
    model = SVD()
    model.fit(trainset)

    # Save the model
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_file}")
