from argparse import ArgumentParser
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-1')

def inspect_data(data):
    print(data.columns)
    print(data.dtypes)
    print(data.head())

def preprocess_data(data, is_training=True):
    print(f"Initial data shape: {data.shape}")

    # Separate trip_id_unique_station to handle it independently
    trip_id_unique_station = data['trip_id_unique_station'].copy()

    # Drop specified columns
    columns_to_drop = ['line_id', 'part', 'trip_id_unique', 'station_id', 'station_name']
    data.drop(columns=columns_to_drop + ['trip_id_unique_station'], inplace=True)

    # Modify specified columns
    data['direction'] = LabelEncoder().fit_transform(data['direction'])
    data['cluster'] = LabelEncoder().fit_transform(data['cluster'])

    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S', errors='coerce')
    data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], format='%H:%M:%S', errors='coerce')

    # Fill missing door_closing_time with the average of existing values
    average_door_closing_time = data['door_closing_time'].dropna().mean()
    data['door_closing_time'].fillna(average_door_closing_time, inplace=True)

    # Remove rows where door_closing_time is before arrival_time
    valid_rows = data['door_closing_time'] >= data['arrival_time']
    data = data[valid_rows]
    trip_id_unique_station = trip_id_unique_station[valid_rows]

    # Create time_in_station
    data['time_in_station'] = (data['door_closing_time'] - data['arrival_time']).dt.total_seconds()
    average_time_in_station = data['time_in_station'].dropna().mean()
    data['time_in_station'].fillna(average_time_in_station, inplace=True)
    data.drop(columns=['arrival_time', 'door_closing_time'], inplace=True)

    data['arrival_is_estimated'] = data['arrival_is_estimated'].astype(int)

    if is_training:
        data['bus_capacity_at_arrival'] = data['passengers_continue'] + data['passengers_up']
    else:
        data['bus_capacity_at_arrival'] = data['passengers_continue']  # No passengers_up in test set

    data.drop(columns=['passengers_continue'], inplace=True)

    # Replace non-numeric values with NaN
    data.replace('#', np.nan, inplace=True)

    print(f"Data shape after dropping columns and replacing values: {data.shape}")

    # Convert all columns to numeric, forcing errors to NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    print(f"Data shape after converting to numeric: {data.shape}")

    # Handle missing values by filling with the median value of the column
    data.fillna(data.median(), inplace=True)

    print(f"Data shape after filling NaN values: {data.shape}")

    # Ensure there are no NaN or infinite values left
    if data.isnull().values.any() or np.isinf(data.values).any():
        print("Warning: NaN or infinity values found in the data after filling NaN values.")
        data = data.dropna()  # Drop rows with remaining NaN or infinity values
        trip_id_unique_station = trip_id_unique_station[data.index]

    # Feature Scaling
    numerical_features = ['station_index', 'latitude', 'longitude', 'mekadem_nipuach_luz',
                          'passengers_continue_menupach', 'time_in_station', 'bus_capacity_at_arrival']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    print(f"Data shape after scaling: {data.shape}")

    if is_training:
        X = data.drop(columns=['passengers_up'])
        y = data['passengers_up']
        return X, y, trip_id_unique_station
    else:
        return data, trip_id_unique_station

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    predictions = model.predict(X)
    return np.clip(predictions, 0, None)  # Ensure no negative predictions

def save_predictions(predictions, output_path, trip_id_unique_station):
    # Round predictions to the nearest whole number
    rounded_predictions = np.round(predictions).astype(int)
    output = pd.DataFrame({'trip_id_unique_station': trip_id_unique_station, 'passengers_up': rounded_predictions})
    output.to_csv(output_path, index=False)

def plot_relationships(data, predictions, features, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    data['passengers_up_predicted'] = predictions

    for feature in features:
        plt.figure(figsize=(10, 6))
        plt.scatter(data[feature], data['passengers_up_predicted'], alpha=0.5)
        plt.title(f'Relationship between {feature} and passengers_up')
        plt.xlabel(feature)
        plt.ylabel('passengers_up')
        plt.grid(True)
        plt.savefig(f"{output_dir}/{feature}_vs_passengers_up.png")
        plt.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True, help="path of the output file as required in the task description")
    parser.add_argument('--plot_dir', type=str, required=True, help="directory to save plots")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    train_data = load_data(args.training_set)

    # Inspect the data
    inspect_data(train_data)

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    X_train, y_train, trip_id_unique_station_train = preprocess_data(train_data, is_training=True)

    # 3. train a model
    logging.info("training...")
    model = train_model(X_train, y_train)

    # 4. load the test set (args.test_set)
    test_data = load_data(args.test_set)

    # Inspect the test data
    inspect_data(test_data)

    # 5. preprocess the test set
    logging.info("preprocessing test...")
    X_test, trip_id_unique_station_test = preprocess_data(test_data, is_training=False)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = predict(model, X_test)

    # Debugging step: Print lengths to ensure they match
    print(f"Length of predictions: {len(predictions)}")
    print(f"Length of trip_id_unique_station_test: {len(trip_id_unique_station_test)}")

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    save_predictions(predictions, args.out, trip_id_unique_station_test)

    # 8. plot relationships between features and passengers_up
    features_to_plot = ['bus_capacity_at_arrival', 'time_in_station', 'cluster', 'direction']
    plot_relationships(test_data, predictions, features_to_plot, args.plot_dir)
