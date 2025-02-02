import pandas as pd
import matplotlib.pyplot as plt
import json

def plot_linear_regression():
    """Plot data and result into a graph"""
    theta0 = 0
    theta1 = 0
    df = pd.read_csv('./data.csv')
    with open("training_results.json", "r") as file:
        tr = json.load(file)
        theta0 = tr['theta0']
        theta1 = tr['theta1']
    plt.scatter(df['km'], df['price'])
    plt.plot(df['km'], theta1 * df['km'] + theta0, color='red')
    plt.title("Linear Regression Fit on Original Scale")
    plt.xlabel("km")
    plt.ylabel("price")
    plt.show()




def main():
    plot_linear_regression()


if __name__ == "__main__":
    main()
