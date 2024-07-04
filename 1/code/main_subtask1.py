from argparse import ArgumentParser
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-8')


def inspect_data(data):
    print(data.columns)
    print(data.dtypes)
    print(data.head())


def preprocess_data(data, is_training=True):
    print(f"Initial data shape: {data.shape}")

    # Separate trip_id_unique_station to handle it independently
    trip_id_unique_station = data['trip_id_unique_station'].copy()
    arrival_time = data['arrival_time'].copy()  # Preserve arrival_time for later use
    data.drop(columns=['trip_id_unique_station'], inplace=True)

    # Drop specified columns
    columns_to_drop = ['line_id', 'part', 'trip_id_unique', 'station_id', 'station_name']
    data.drop(columns=columns_to_drop, inplace=True)

    # Modify specified columns
    data['direction'] = LabelEncoder().fit_transform(data['direction'])
    data['cluster'] = LabelEncoder().fit_transform(data['cluster'])

    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S', errors='coerce')
    data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], format='%H:%M:%S', errors='coerce')
    data['time_in_station'] = (data['door_closing_time'] - data['arrival_time']).dt.total_seconds()
    data.drop(columns=['arrival_time', 'door_closing_time'], inplace=True)

    data['arrival_is_estimated'] = data['arrival_is_estimated'].astype(int)

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

    # Feature Scaling
    numerical_features = ['station_index', 'latitude', 'longitude', 'mekadem_nipuach_luz',
                          'passengers_continue_menupach', 'time_in_station', 'passengers_continue']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    print(f"Data shape after scaling: {data.shape}")

    if is_training:
        X = data.drop(columns=['passengers_up'])
        y = data['passengers_up']
        return X, y, trip_id_unique_station, arrival_time
    else:
        return data, trip_id_unique_station, arrival_time


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
    return rounded_predictions


def plot_relationships(data, predictions, features, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    data['passengers_up_predicted'] = predictions

    # Debugging step: Print columns to ensure they exist
    print("Columns available for plotting:", data.columns)

    for feature in features:
        if feature in data.columns:
            plt.figure(figsize=(10, 6))
            if feature == "time_in_station":
                # Exclude negative values and keep only those less than k
                k = 5
                filtered_data = data[(data['time_in_station'] >= 0) & (data['time_in_station'] < k)]
                plt.scatter(filtered_data[feature], filtered_data['passengers_up_predicted'], alpha=0.5)
                plt.xlabel(f'{feature} (minutes)')
            else:
                plt.scatter(data[feature], data['passengers_up_predicted'], alpha=0.5)
                plt.xlabel(feature)
            plt.title(f'Relationship between {feature} and passengers_up')
            plt.ylabel('passengers_up')
            plt.grid(True)
            plt.savefig(f"{output_dir}/{feature}_vs_passengers_up.png")
            plt.close()
        else:
            print(f"Feature {feature} not found in data columns.")


def generate_additional_plots(data, arrival_time, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    # Convert 'arrival_time' back to datetime for plotting if necessary
    data['arrival_time'] = pd.to_datetime(arrival_time, format='%H:%M:%S', errors='coerce')
    # data['arrival_time'] = arrival_time

    # Plot 1: Peak Rush Hours Analysis
    data['hour'] = data['arrival_time'].dt.hour
    plt.figure(figsize=(12, 6))
    sns.histplot(data['hour'], bins=24, kde=False)
    plt.title('Number of Passengers Boarding at Different Times of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Passengers Boarding')
    plt.grid(True)
    plt.savefig(f"{output_dir}/peak_rush_hours.png")
    plt.close()

    # # Plot 2: Heatmap of Bus Usage Across Clusters Over Time
    # data['hour'] = data['arrival_time'].dt.hour
    # cluster_hourly_data = data.groupby(['cluster', 'hour']).agg({'passengers_up': 'mean'}).reset_index()
    # cluster_hourly_pivot = cluster_hourly_data.pivot('cluster', 'hour', 'passengers_up')
    #
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(cluster_hourly_pivot, cmap="YlGnBu", annot=True, fmt=".1f", linewidths=.5)
    # plt.title('Heatmap of Average Bus Usage Across Clusters Over Time')
    # plt.xlabel('Hour of the Day')
    # plt.ylabel('Cluster')
    # plt.savefig(f"{output_dir}/heatmap_bus_usage_across_clusters.png")
    # plt.close()

    # Plot 3: Consistency in Public Transportation Usage (daily usage)
    data['day'] = data['arrival_time'].dt.day_name()
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='day', y='passengers_up', data=data)
    plt.title('Consistency in Public Transportation Usage (Daily)')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Passengers Boarding')
    plt.grid(True)
    plt.savefig(f"{output_dir}/consistency_daily_usage.png")
    plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    parser.add_argument('--plot_dir', type=str, required=True, help="directory to save plots")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    train_data = load_data(args.training_set)

    # Inspect the data
    inspect_data(train_data)

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    X_train, y_train, trip_id_unique_station_train, arrival_time_train = preprocess_data(train_data, is_training=True)

    # 3. train a model
    logging.info("training...")
    model = train_model(X_train, y_train)

    # 4. load the test set (args.test_set)
    test_data = load_data(args.test_set)

    # Inspect the test data
    inspect_data(test_data)

    # 5. preprocess the test set
    logging.info("preprocessing test...")
    X_test, trip_id_unique_station_test, arrival_time_test = preprocess_data(test_data, is_training=False)
    # Debugging step: Ensure all columns are numeric
    print("Columns before prediction:", X_test.columns)
    print("Data types before prediction:", X_test.dtypes)

    # Remove trip_id_unique_station from the test data before prediction, if it exists
    if 'trip_id_unique_station' in X_test.columns:
        X_test = X_test.drop(columns=['trip_id_unique_station'])

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = predict(model, X_test)

    # Debugging step: Print lengths to ensure they match
    print(f"Length of predictions: {len(predictions)}")
    print(f"Length of trip_id_unique_station_test: {len(trip_id_unique_station_test)}")

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    rounded_predictions = save_predictions(predictions, args.out, trip_id_unique_station_test)

    # Add passengers up
    test_data["passengers_up"] = rounded_predictions

    # 8. plot relationships between features and passengers_up
    features_to_plot = ['passengers_continue', 'time_in_station', 'cluster', 'direction']
    plot_relationships(X_test, predictions, features_to_plot, args.plot_dir)

    # 9. generate additional plots for insights
    logging.info("generating additional plots...")
    generate_additional_plots(test_data, arrival_time_test, args.plot_dir)
