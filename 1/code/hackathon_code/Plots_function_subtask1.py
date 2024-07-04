import plotly as plt
import syslog as sns
import os
import pandas as pd

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
