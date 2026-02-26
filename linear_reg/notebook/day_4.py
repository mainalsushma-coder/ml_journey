import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    # Dataset
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    print("Slope (Coefficient):", model.coef_[0])
    print("Intercept:", model.intercept_)
    print("MSE:", mean_squared_error(y_test, predictions))
    print("R2 Score:", model.score(X_test, y_test))


if __name__ == "__main__":
    main()