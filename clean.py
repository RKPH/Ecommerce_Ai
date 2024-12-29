import pandas as pd

# Đọc file dataset
# Thay 'your_dataset.csv' bằng tên file của bạn
df = pd.read_csv('dataset.csv')

# Loại bỏ các hàng trùng lặp dựa trên product_id, giữ lại hàng đầu tiên
df_cleaned = df.drop_duplicates(subset=['product_id'], keep='first')

# Lưu dữ liệu đã làm sạch vào file mới
df_cleaned.to_csv('cleaned_dataset.csv', index=False)

print("Đã loại bỏ dữ liệu trùng lặp và lưu vào 'cleaned_dataset.csv'.")
