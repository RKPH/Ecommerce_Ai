import pandas as pd

# Đọc file dataset (thay 'your_dataset.csv' bằng tên file của bạn)
df = pd.read_csv('../Data/3_session_dataset_5M.csv')  # Đảm bảo rằng file CSV tồn tại và đường dẫn chính xác

# Loại bỏ các hàng có bất kỳ giá trị NaN nào
df_cleaned = df.dropna()

# Lưu dữ liệu đã làm sạch vào file mới
df_cleaned.to_csv('Data/dataset.csv', index=False)

print("Đã loại bỏ các hàng có giá trị trống, lưu vào 'dataset.csv'.")
