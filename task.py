import logging
from argparse import ArgumentParser
import  os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from geopy.distance import geodesic
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 
"""
def plot_mse_vs_percentage(results):
    percentages, mse_values = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, mse_values, marker='o')
    plt.title('MSE vs. Percentage of Training Data')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xticks(percentages)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_mse_percentage(train_data, test_data, percentages=list(range(5,100,5))):
    results = []
    for percent in percentages:
        # Sample the training data based on the percentage
        train_subset = train_data.sample(frac=percent / 100, random_state=42)

        # Separate features and target variable
        x_train = train_subset.drop(['trip_duration'], axis=1)
        y_train = train_subset['trip_duration']

        # Train the model
        model = train_model(x_train, y_train)

        # Preprocess test data

        x_test = test_data.drop(['trip_duration'], axis=1)

        # Predict using the model
        predictions = predict(model, x_test)

        # Calculate MSE
        mse = mean_squared_error(test_data['trip_duration'], predictions)
        results.append((percent, mse))

    return results

# implement here your load, preprocess, train, predict, save functions (or any other design you choose)
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-8')
    return data


def preprocess_train(data):
    # Drop rows with missing values
    data = data.dropna()

    # Convert 'arrival_time' to datetime
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S')

    # Ensure data is sorted by 'trip_id_unique', 'direction', and 'station_index'
    data = data.sort_values(by=['trip_id_unique', 'direction', 'station_index'])

    # Group by 'trip_id_unique' and calculate the time difference between the first and last station
    trip_time_diff = data.groupby(['trip_id_unique'])['arrival_time'].agg(
        ['first', 'last'])

    # Step 1: Create a mapping of cluster names to numeric labels
    trip_time_diff['cluster'] = (data.groupby('trip_id_unique')['cluster'].agg(['first']).transform(lambda x: pd.factorize(x)[0]))
    # trip_time_diff['cluster'] = .transform(lambda x: pd.factorize(x)[0])

    total_passengers_per_trip = data.groupby('trip_id_unique')['passengers_up'].sum().astype(int)

    station_cor = data.groupby(['trip_id_unique'])[['latitude', 'longitude']].agg(
        ['first', 'last']).astype(float)

    # Calculate trip duration considering potential day change
    trip_time_diff['trip_duration'] = trip_time_diff.apply(
        lambda x: (x['last'] - x['first']).total_seconds() / 60.0
        if x['last'] >= x['first']
        else ((pd.Timedelta(days=1) + x['last'] - x['first']).total_seconds() / 60.0),
        axis=1)

    # Remove rows where trip duration is negative
    trip_time_diff = trip_time_diff[trip_time_diff['trip_duration'] >= 0]

    # Extract the start hour and minute for each trip
    trip_time_diff['start_hour'] = trip_time_diff['first'].dt.hour.astype(int)

    # Merge trip_time_diff back into original data based on 'trip_id_unique'
    trip_time_diff = pd.merge(total_passengers_per_trip,
                              trip_time_diff,
                              left_on='trip_id_unique', right_index=True, how='left')

    # Calculate distance between first and last station based on latitude and longitude
    def calculate_distance(row):
        lat_first = row[('latitude', 'first')]
        lat_last = row[('latitude', 'last')]
        lon_first = row[('longitude', 'first')]
        lon_last = row[('longitude', 'last')]
        coords1 = (lat_first, lon_first)
        coords2 = (lat_last, lon_last)
        return geodesic(coords1, coords2).km  # Using geodesic for distance calculation

    trip_dist = station_cor.apply(calculate_distance, axis=1).astype(float)
    trip_dist.name = 'trip_dist'

    # Calculate distances using geopy
    trip_time_diff = pd.merge(trip_time_diff,
                              trip_dist,
                              left_on='trip_id_unique', right_index=True, how='left')

    return trip_time_diff.drop(['first', 'last'], axis=1)


def preprocess_test(data):
    data = data.dropna()
    return data


def train_model(x_train, y_train):
    model = DecisionTreeRegressor(max_depth=12)
    model.fit(x_train, y_train)
    return model


def predict(model, x_test):
    return model.predict(x_test)


def save_predictions(predictions, output_path):
    os.makedirs('predictions', exist_ok=True)
    predictions.to_csv(output_path)


def visualize_data(data):
    # Plot trip duration vs. start hour
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='start_hour', y='trip_duration', data=data)
    plt.title('Trip Duration vs. Start Hour')
    plt.xlabel('Start Hour of Trip')
    plt.ylabel('Trip Duration (minutes)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    logging.info("loading training set...")
    train_data = load_data(args.training_set)

    # 2. preprocess the training set
    logging.info("preprocessing training set...")
    train_data_processed = preprocess_train(train_data)

    # Visualize data
    logging.info("visualizing data...")
    visualize_data(train_data_processed)

    # Separate features and target variable
    x_train = train_data_processed.drop(['trip_duration'], axis=1)
    y_train = train_data_processed['trip_duration']

    # 3. train a model
    logging.info("training model...")
    model = train_model(x_train, y_train)

    # 4. load the test set (args.test_set)
    logging.info("loading test set...")
    test_data = load_data(args.test_set)

    # 5. preprocess the test set
    logging.info("preprocessing test set...")
    test_data_processed = preprocess_train(test_data)

    # Prepare test data for prediction
    x_test = test_data_processed.drop(['trip_duration'], axis=1)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = predict(model, x_test)

    # Combine predictions with trip IDs
    prediction_results = pd.DataFrame(
        {'trip_id_unique': x_test.index, 'predicted_duration': predictions})

    # 7. save the predictions to args.out
    logging.info(f"saving predictions to {args.out}...")
    save_predictions(prediction_results, args.out)
    logging.info("predictions saved successfully.")
    mse_results = calculate_mse_percentage(train_data_processed, test_data_processed)

    # Print the results

    plot_mse_vs_percentage(mse_results)