# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Tùy chọn, dùng để vẽ biểu đồ nâng cao
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import folium
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# BƯỚC 1: Đọc dữ liệu và tiền xử lý
# =============================================================================

# Đọc dữ liệu từ file CSV
df = pd.read_csv('data_onkk.csv')

# In thông tin ban đầu để kiểm tra cấu trúc dữ liệu
print("Thông tin dữ liệu ban đầu:")
print(df.head())
print("\nDanh sách các cột:", df.columns.tolist())

# Hàm tính AQI từ nồng độ PM2.5 theo tiêu chuẩn US EPA
def compute_aqi_pm25(concentration):
    """
    Tính chỉ số AQI từ nồng độ PM2.5 theo các ngưỡng của US EPA.
    """
    if concentration < 0:
        return np.nan
    elif concentration <= 12.0:
        aqi = (50 / 12.0) * concentration
    elif concentration <= 35.4:
        aqi = ((100 - 51) / (35.4 - 12.1)) * (concentration - 12.1) + 51
    elif concentration <= 55.4:
        aqi = ((150 - 101) / (55.4 - 35.5)) * (concentration - 35.5) + 101
    elif concentration <= 150.4:
        aqi = ((200 - 151) / (150.4 - 55.5)) * (concentration - 55.5) + 151
    elif concentration <= 250.4:
        aqi = ((300 - 201) / (250.4 - 150.5)) * (concentration - 150.5) + 201
    elif concentration <= 350.4:
        aqi = ((400 - 301) / (350.4 - 250.5)) * (concentration - 250.5) + 301
    elif concentration <= 500.4:
        aqi = ((500 - 401) / (500.4 - 350.5)) * (concentration - 350.5) + 401
    else:
        aqi = np.nan
    return aqi

# Tính AQI dựa trên cột "pm25"
if 'pm25' in df.columns:
    df['AQI'] = df['pm25'].apply(compute_aqi_pm25)
else:
    print("Cột 'pm25' không tồn tại, vui lòng kiểm tra lại file dữ liệu.")

# Chuyển đổi cột "time" sang kiểu datetime nếu có
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.sort_values('time')

# Hiển thị vài dòng dữ liệu sau khi tính toán AQI
print("\nDữ liệu sau khi tính toán AQI:")
print(df.head())

# =============================================================================
# BƯỚC 2: Khám phá dữ liệu (EDA)
# =============================================================================

if 'time' in df.columns:
    plt.figure(figsize=(12,6))
    plt.plot(df['time'], df['pm25'], label='PM2.5')
    plt.plot(df['time'], df['AQI'], label='AQI')
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.title("Biểu đồ PM2.5 và AQI theo thời gian")
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# BƯỚC 3: Xây dựng và đánh giá mô hình dự báo AQI
# =============================================================================

# Xác định các biến không dùng cho mô hình: time, ID, pm25, AQI, lat, lon
cols_to_exclude = []
for col in ['time', 'ID', 'pm25', 'AQI', 'lat', 'lon']:
    if col in df.columns:
        cols_to_exclude.append(col)

# Các biến đặc trưng (features) là các cột còn lại
features = [col for col in df.columns if col not in cols_to_exclude]
print("\nCác biến đặc trưng được sử dụng cho mô hình:", features)

# Loại bỏ các dòng có giá trị thiếu ở cột AQI và các biến đặc trưng
df_model = df.dropna(subset=['AQI'] + features)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra theo thời gian nếu có cột "time"
if 'time' in df_model.columns:
    cutoff_time = df_model['time'].quantile(0.8)  # 80% dữ liệu đầu làm train
    train_data = df_model[df_model['time'] < cutoff_time]
    test_data = df_model[df_model['time'] >= cutoff_time]
else:
    train_data, test_data = train_test_split(df_model, test_size=0.2, random_state=42)

# Xác định X và y cho train và test
X_train = train_data[features]
y_train = train_data['AQI']
X_test = test_data[features]
y_test = test_data['AQI']

print("\nSố lượng mẫu - Tập huấn luyện: {}, Tập kiểm tra: {}".format(X_train.shape[0], X_test.shape[0]))

# --- Mô hình 1: Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
pred_lr = lr_model.predict(X_test)

# Đánh giá Linear Regression
mae_lr = mean_absolute_error(y_test, pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
r2_lr = r2_score(y_test, pred_lr)

print("\nKết quả mô hình Linear Regression:")
print("MAE: {:.2f}".format(mae_lr))
print("RMSE: {:.2f}".format(rmse_lr))
print("R²: {:.2f}".format(r2_lr))

# --- Mô hình 2: Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

# Đánh giá Random Forest
mae_rf = mean_absolute_error(y_test, pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
r2_rf = r2_score(y_test, pred_rf)

print("\nKết quả mô hình Random Forest:")
print("MAE: {:.2f}".format(mae_rf))
print("RMSE: {:.2f}".format(rmse_rf))
print("R²: {:.2f}".format(r2_rf))

# Chọn mô hình có hiệu năng tốt hơn (ở đây dựa trên R²; bạn có thể kết hợp thêm các chỉ số khác)
if r2_rf >= r2_lr:
    best_model = rf_model
    best_model_name = "Random Forest"
else:
    best_model = lr_model
    best_model_name = "Linear Regression"

print("\nMô hình tốt nhất được chọn:", best_model_name)

# =============================================================================
# BƯỚC 4: Dự báo AQI cho ngày tiếp theo
# =============================================================================
# Lưu ý: Trong thực tế, để dự báo cho ngày tiếp theo bạn cần có dự báo các thông số đầu vào (khí tượng, địa hình) cho ngày đó.
# Ở đây, để minh họa, ta sử dụng mẫu cuối cùng của dữ liệu như một đầu vào giả lập cho ngày tiếp theo.
last_record = df_model.iloc[-1]
X_new = last_record[features].values.reshape(1, -1)
forecast_aqi = best_model.predict(X_new)[0]

print("\nDự báo AQI cho ngày tiếp theo (theo mô hình {}): {:.2f}".format(best_model_name, forecast_aqi))

# =============================================================================
# BƯỚC 5: Hiển thị bản đồ dự báo (sử dụng Folium)
# =============================================================================

if 'lat' in df.columns and 'lon' in df.columns:
    # Chọn tọa độ trung tâm cho miền Bắc Việt Nam (ví dụ: Hà Nội)
    map_center = [21.028511, 105.804817]
    aqi_map = folium.Map(location=map_center, zoom_start=7)

    # Thêm marker cho từng trạm với thông tin AQI, thời gian và ID
    for idx, row in df.iterrows():
        try:
            lat_val = float(row['lat'])
            lon_val = float(row['lon'])
            aqi_val = row['AQI']
            popup_text = f"ID: {row['ID']}<br>Time: {row['time']}<br>AQI: {aqi_val:.2f}"
            folium.CircleMarker(location=[lat_val, lon_val],
                                radius=5,
                                popup=popup_text,
                                fill=True,
                                color='blue',
                                fill_opacity=0.7).add_to(aqi_map)
        except Exception as e:
            continue

    # Lưu bản đồ dưới dạng file HTML
    aqi_map.save('forecast_map.html')
    print("\nBản đồ dự báo đã được lưu vào file 'forecast_map.html'. Mở file này trong trình duyệt để xem kết quả.")
else:
    print("\nKhông có dữ liệu tọa độ (lat, lon) để hiển thị bản đồ.")
