# Air Quality Index (AQI) Forecasting for Northern Vietnam

## Contributors
Nguyễn Quang Cảnh - 22028200

Nguyễn Văn Thuận - 22028194


##  Overview
This project aims to forecast the Air Quality Index (AQI) based on PM2.5 concentration data and additional meteorological and topographical inputs. In this implementation, we compute AQI using the EPA standard from PM2.5 measurements, build a simple neural network model to predict AQI, and visualize both the data distributions and model performance. The primary study area is Northern Vietnam.

##  Project Description
The goal of this project is to:
+ **Calculate AQI**: Compute the AQI from PM2.5 readings using breakpoints as defined by the EPA.
+ **Experiment with Models**: Test a neural network-based machine learning model on the dataset.
+ **Evaluate Model Performance**: Assess the model using evaluation metrics such as Mean Squared Error (MSE) and R².
+ **Visualize Results**: Create plots to show the distribution of PM2.5, distribution of AQI, scatter plots to illustrate the relationship between PM2.5 and AQI, and a loss (error) curve across training epochs.
+ **Provide a Basis for Future Mapping**: Although this implementation focuses on the modeling and evaluation, it serves as a foundation for developing forecast maps for PM2.5 and AQI.

##  Dataset
The dataset is provided in a CSV file `data_onkk.csv` with 11,508 entries and 14 columns. Key columns include:

+ **time**: Timestamp for each record.
+ **pm25**: Measured PM2.5 concentration.
+ **lat & lon**: Geographic coordinates.
+ **Other meteorological and topographical data**:(e.g., temperature, wind speed, elevation, etc.)
For the purposes of this project, the focus is on the `pm25` column. The AQI is computed from this column using the EPA breakpoints.

##  Data Preprocessing
1. **Data Loading & Exploration**:
+ The CSV file is read into a pandas DataFrame.
+ Basic information, a sample of the first five rows, and summary statistics are printed to understand the data structure.
+ Missing values are checked to ensure data integrity.
2. **AQI Calculation**:
+ A function `compute_aqi` is defined to calculate the AQI based on PM2.5 concentrations using the EPA standard breakpoints.
+ The computed AQI is added as a new column in the DataFrame.
3. **Visualization of Data Distributions**:
+ Histograms are plotted for PM2.5 and the computed AQI.
+ A scatter plot is created to show the relationship between PM2.5 and AQI.

##  Model Building & Training
1. **Dataset Splitting**:
+ The data is split into training (80%) and testing (20%) sets using `train_test_split` from scikit-learn.
2. **Neural Network Model**:
A simple neural network is built using TensorFlow/Keras.

The network architecture includes:
+ **Input layer**: Receives the PM2.5 value (1 feature).
+ **First hidden layer**: 16 neurons with ReLU activation.
+ **Second hidden layer**: 8 neurons with ReLU activation.
+ **Output layer**: 1 neuron to predict the AQI.
The model is compiled with the Adam optimizer and Mean Squared Error (MSE) as the loss function.

The model is trained for 200 epochs on the training set with a validation split (20% of training data) to monitor performance.

## Model Evaluation & Visualization
1. **Performance Metrics**:
+ The model is evaluated on the test set using MSE and R².
+ These metrics help gauge how well the model predicts AQI from PM2.5 concentrations.
2. **Visualization of Predictions**:
+ Predictions on the test set and overall dataset are generated.
+ A comparison plot is created showing the true AQI values against the predicted AQI values, sorted by PM2.5 for clarity.
+ A loss curve is plotted across epochs to display the training and validation loss trends.

## How to Run the Project
1. **Prerequisites**: 
Ensure you have Python 3.x installed and the following packages:
+ pandas
+ numpy
+ matplotlib
+ seaborn
+ scikit-learn
+ tensorflow
You can install the dependencies using pip:

`pip install pandas numpy matplotlib seaborn scikit-learn tensorflow`

2. **Running the Code**:
**Step 1**: Prepare the dataset
Ensure that the dataset file `data_onkk.csv` is placed in the correct directory as referenced in the script (e.g., `data_onkk.csv`).

**Step 2**: Run the script
+ *On Windows (Command Prompt or PowerShell)*: 
`python air_quality_forecast.py`
+ *On macOS & Linux (Terminal)*: 
`python3 air_quality_forecast.py`

(If python3 is not recognized, use python instead.)

**The script will**:
+ Load and analyze the data.
+ Compute AQI values from PM2.5 concentrations.
+ Split the dataset into training (80%) and testing (20%).
+ Train the neural network model.
+ Evaluate model performance.
+ Generate and display plots, including data distributions, + scatter plots, and training loss trends.
## File Structure
+ `air_quality_forecast.py`: Main Python script containing the model code 
+ `data_onkk.csv`: Input dataset file 
+ **PM2.5_AQI_Distribution.png**: Histogram(s) of PM2.5 and/or AQI distribution 
+ **PM2.5_AQI_ScatterPlot.png**: Scatter plot showing relationship between PM2.5 and AQI
+ **Actual_Predicted_AQI.png**: Visualization of actual vs. predicted AQI 
+ **Loss_over_epochs.png**: Visualization of loss over training epochs 

## Conclusion
This project demonstrates how to preprocess environmental data, compute derived indices (AQI from PM2.5), build and train a neural network model, and visualize the results. The implementation provides a basis for further development, including advanced forecasting methods and geographic mapping for air quality monitoring.

