from argparse import ArgumentParser
import logging


"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""

# implement here your load,preprocess,train,predict,save functions (or any other design you choose)
from argparse import ArgumentParser
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-8')


def preprocess_data(data, is_training=True):
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
    data['bus_capacity_at_arrival'] = data['passengers_continue'] + data['passengers_up']
    data.drop(columns=['passengers_continue'], inplace=True)

    # Feature Scaling
    numerical_features = ['station_index', 'latitude', 'longitude', 'mekadem_nipuach_luz',
                          'passengers_continue_menupach', 'time_in_station', 'bus_capacity_at_arrival']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    if is_training:
        X = data.drop(columns=['passengers_up'])
        y = data['passengers_up']
        return X, y
    else:
        return data


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    return model.predict(X)


def save_predictions(predictions, output_path):
    predictions.to_csv(output_path, index=False)


def evaluate_model(X_train, X_val, y_train, y_val, features):
    model = LinearRegression()
    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_val[features])
    mse = mean_squared_error(y_val, y_pred)
    return mse


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    train_data = load_data(args.training_set)

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    X_train, y_train = preprocess_data(train_data, is_training=True)

    # 3. train a model
    logging.info("training...")
    model = train_model(X_train, y_train)

    # 4. load the test set (args.test_set)
    test_data = load_data(args.test_set)

    # 5. preprocess the test set
    logging.info("preprocessing test...")
    X_test = preprocess_data(test_data, is_training=False)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = predict(model, X_test)

    # Prepare the output in the required format
    output = pd.DataFrame({'trip_id_unique_station': test_data['trip_id_unique_station'], 'passengers_up': predictions})

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    save_predictions(output, args.out)

#
# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--training_set', type=str, required=True,
#                         help="path to the training set")
#     parser.add_argument('--test_set', type=str, required=True,
#                         help="path to the test set")
#     parser.add_argument('--out', type=str, required=True,
#                         help="path of the output file as required in the task description")
#     args = parser.parse_args()
#
#     # 1. load the training set (args.training_set)
#     # 2. preprocess the training set
#     logging.info("preprocessing train...")
#
#     # 3. train a model
#     logging.info("training...")
#
#     # 4. load the test set (args.test_set)
#     # 5. preprocess the test set
#     logging.info("preprocessing test...")
#
#     # 6. predict the test set using the trained model
#     logging.info("predicting...")
#
#     # 7. save the predictions to args.out
#     logging.info("predictions saved to {}".format(args.out))
