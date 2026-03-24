import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- BƯỚC 1: CHUẨN BỊ DỮ LIỆU ---
# (Dữ liệu đầu vào từ danh sách bạn cung cấp)
# Giả sử dữ liệu đã được lưu vào file iris.csv
try:
    df = pd.read_csv('iris.csv')
except:
    # Nếu chưa có file, ta có thể dùng trực tiếp dữ liệu từ sklearn để mô phỏng
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
    df['variety'] = iris_data.target

# Tách đặc trưng (X) và nhãn (y)
X = df.drop('variety', axis=1)
y = df['variety']

# --- BƯỚC 2: TÁCH TẬP TRAIN VÀ TEST (TỶ LỆ 8:2) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH ---
# Chọn k = 3 (3 láng giềng gần nhất)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# --- BƯỚC 4: KIỂM THỬ VÀ ĐÁNH GIÁ ---
y_pred = knn.predict(X_test)

# In kết quả
print(f"Độ chính xác tổng thể: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nBáo cáo chi tiết hiệu suất:")
print(classification_report(y_test, y_pred))