import pandas as pd
import json


def caculate_precision():
    """Compute RMSE (Root Mean Squared Error)"""
    theta0 = 0
    theta1 = 0
    df = pd.read_csv("./data/data.csv")

    X = df["km"]
    Y = df["price"]
    n = float(len(X))
    with open(".training_results.json", "r") as file:
        tr = json.load(file)
        theta0 = tr["theta0"]
        theta1 = tr["theta1"]
    rmse = sum(1 / n * (Y - (X * theta1 + theta0)) ** 2) ** (1 / 2)
    print(f"RMSE: {rmse}")


def main():
    caculate_precision()


if __name__ == "__main__":
    main()
