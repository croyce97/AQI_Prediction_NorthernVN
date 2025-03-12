# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ---------------------------
# 1. Read and analyze data
# ---------------------------
# Read data from CSV file
data = pd.read_csv('data_onkk.csv')

# Print data information
print("Data information:")
print(data.info())

# Print the first 5 rows of the dataset
print("\nFirst 5 rows of data:")
print(data.head())

# Descriptive statistics of the dataset
print("\nDescriptive statistics:")
print(data.describe())

# Check the number of missing values in each column
print("\nNumber of missing values:")
print(data.isnull().sum())

# ---------------------------
# 2. Calculate AQI based on PM2.5 concentration
# ---------------------------
# Function to calculate AQI based on EPA standards using PM2.5 concentration
def compute_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= pm25 <= C_high:
            aqi = ((I_high - I_low) / (C_high - C_low)) * (pm25 - C_low) + I_low
            return aqi
    return np.nan  # Return NaN if PM2.5 is out of range

# Compute the AQI column based on the 'pm25' column
data['AQI'] = data['pm25'].apply(compute_aqi)

# Check descriptive statistics for the AQI column
print("\nDescriptive statistics for AQI column:")
print(data['AQI'].describe())

# ---------------------------
# 3. Visualize the initial data
# ---------------------------
# Plot distribution of PM2.5 and AQI
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(data['pm25'], bins=30, kde=True)
plt.title('PM2.5 distribution')

plt.subplot(1, 2, 2)
sns.histplot(data['AQI'], bins=30, kde=True)
plt.title('AQI distribution')

plt.tight_layout()
plt.show()

# Scatter plot of PM2.5 vs AQI
plt.figure(figsize=(7, 6))
sns.scatterplot(x='pm25', y='AQI', data=data)
plt.xlabel('PM2.5')
plt.ylabel('AQI')
plt.title('Scatter plot: PM2.5 vs AQI')
plt.show()

# ---------------------------
# 4. Prepare data and split into training (80%) and testing (20%) sets
# ---------------------------
# Drop rows with missing values in the AQI column (if any)
data = data.dropna(subset=['AQI'])

# Define input variable X (PM2.5) and target variable y (AQI)
X = data['pm25'].values.reshape(-1, 1)
y = data['AQI'].values

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# ---------------------------
# 5. Build and train the machine learning model
# ---------------------------
# Build a neural network model with 2 hidden layers
model = Sequential([
    Dense(16, activation='relu', input_shape=(1,)),
    Dense(8, activation='relu'),
    Dense(1)  # Output: Predicted AQI value
])

# Compile the model with MSE loss function and Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Train the model on the training set, using 20% of the training set as validation data
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=1)

# ---------------------------
# 6. Evaluate the model on the test set
# ---------------------------
# Predict on the test set
y_test_pred = model.predict(X_test).flatten()

# Compute evaluation metrics
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"\nMSE on test set: {mse_test:.2f}")
print(f"R² on test set: {r2_test:.2f}")

# ---------------------------
# 7. Visualize the results
# ---------------------------
# Predict on the entire dataset for comparison visualization
y_pred = model.predict(X).flatten()

# Sort data by PM2.5 values for better visualization
sorted_idx = X.flatten().argsort()
X_sorted = X.flatten()[sorted_idx]
y_sorted = y[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.figure(figsize=(10, 6))
plt.plot(X_sorted, y_sorted, 'o-', label='AQI (Actual)', markersize=4)
plt.plot(X_sorted, y_pred_sorted, 'r-', label='AQI (Predicted)')
plt.xlabel('PM2.5')
plt.ylabel('AQI')
plt.title('Comparison of actual and predicted AQI (80% training, 20% testing)')
plt.legend()
plt.show()

# Plot Loss over epochs to track training process
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Loss over epochs')
plt.legend()
plt.show()
