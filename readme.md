# 🚗 Linear Regression — Car Price Predictor

This project implements a simple linear regression model to estimate car prices based on mileage. It is split into two main parts: **model training** and **price prediction**.

---

## 📁 Project Structure

- `trainer.py`: Reads a dataset, trains a linear regression model, and saves the parameters.
- `predictor.py`: Prompts the user for mileage input and predicts the car price using the trained model.
- `plotter.py` *(bonus)*: Visualizes the dataset and the regression line.
- `evaluator.py` *(bonus)*: Calculates and displays the precision of the model.
- `data.csv`: Dataset file used for training (mileage vs. price).
- `.training_results.json`: File where trained parameters `theta0` and `theta1` are saved.

---

## 🚀 Getting Started

### 1. Create a Virtual Environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate
