import socket
import pandas as pd
import numpy as np
import json


def train_model(learning_rate=0.01, iterations=1000):
    """Train linear regression using gradient descent with normalized features."""
    df = pd.read_csv("./data/data.csv")

    # Normalize features
    X = (df["km"] - df["km"].mean()) / df["km"].std()
    Y = df["price"]

    # Initialize thetas
    theta0, theta1 = 0, 0

    for i in range(iterations):
        # Predictions
        Y_pred = theta1 * X + theta0

        # Gradient descent update
        tmp_theta0 = learning_rate * np.mean(Y_pred - Y)
        tmp_theta1 = learning_rate * np.mean((Y_pred - Y) * X)

        if abs(tmp_theta0) < 0.001 or abs(tmp_theta1) < 0.001:
            break

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        # Compute true thetas
        theta1_true = theta1 / df["km"].std()
        theta0_true = theta0 - theta1_true * df["km"].mean()

        # Log progress and yield true thetas
        yield theta0_true, theta1_true


def start_trainer(host="127.0.0.1", port=65431):
    """Send updated theta values to the predictor in real-time without blocking training."""
    for iteration, (theta0, theta1) in enumerate(train_model(), start=1):
        if iteration % 100 == 0:  # Send every 100 iterations
            message = f"{theta0},{theta1}"
            try:
                send_update(host, port, message)
                print(f"Sent: theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")
            except (socket.error, ConnectionError) as e:
                print(f"Warning: Unable to send update at iteration {iteration}: {e}")

    message = f"{theta0},{theta1}"
    save_training_results(message)


def send_update(host, port, message):
    """Handles the socket connection and sends theta values."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(message.encode())


def save_training_results(message):
    """Save the final training result to a file."""
    theta0, theta1 = message.split(",")
    with open(".training_results.json", "w") as file:
        json.dump({"theta0": float(theta0), "theta1": float(theta1)}, file, indent=4)


def main():
    start_trainer()


if __name__ == "__main__":
    main()
