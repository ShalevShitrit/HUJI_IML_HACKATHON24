import plotly as plt
import syslog as sns
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
