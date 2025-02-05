import pandas as pd

root = "Data/products.csv"
root_ = "Data/3_session_dataset_5M.csv"

df_hung = pd.read_csv(root)
df_tung = pd.read_csv(root_)

# Ensure column names are correct
df_hung = df_hung[["productID", "name"]].rename(columns={'productID': 'product_id'})

# Perform left join to retain all columns from df_tung
merged_df = df_tung.merge(df_hung, on='product_id', how='left')

# Save CSV while keeping column names
merged_df.to_csv("merged_data_second.csv", index=False, header=True)
