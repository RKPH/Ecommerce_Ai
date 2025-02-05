from Ai_recommendation.process_data.preprocessing import preprocess_data
from Ai_recommendation.Colaborative.train_model import train_model
import os

# Path to CSV dataset
csv_file_path = "Data/dataset.csv"

if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"Dataset file not found at {csv_file_path}")

print("Preprocessing data...")
df, df_weighted = preprocess_data(csv_file_path)

model_file = "Model/model.pkl"

if not os.path.exists(model_file):
    print("Training model...")
    train_model(df_weighted, model_file)
else:
    print(f"Model already exists at {model_file}. Skipping training.")



# User ID input only once
