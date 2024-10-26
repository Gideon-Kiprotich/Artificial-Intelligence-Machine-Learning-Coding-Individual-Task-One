import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Nairobi Office Price Ex.csv'
data = pd.read_csv(file_path)

# Extract the relevant columns: SIZE as the feature (x) and PRICE as the target (y)
x = data['SIZE'].values
y = data['PRICE'].values


# Define the Mean Squared Error (MSE) function
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Define the Gradient Descent function for a simple linear regression model
def gradient_descent(x, y, m, c, learning_rate, epochs):
    n = len(y)
    errors = []

    for epoch in range(epochs):
        # Compute predictions
        y_pred = m * x + c
        # Calculate the error (MSE)
        error = compute_mse(y, y_pred)
        errors.append(error)

        # Calculate gradients
        dm = (-2 / n) * np.sum(x * (y - y_pred))
        dc = (-2 / n) * np.sum(y - y_pred)

        # Update parameters
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Print the error for each epoch
        print(f"Epoch {epoch + 1}/{epochs}, MSE: {error}")

    return m, c, errors


# Initialize parameters
np.random.seed(42)
m = np.random.randn()  # Random slope
c = np.random.randn()  # Random y-intercept
learning_rate = 0.0001
epochs = 10

# Train the model
m, c, errors = gradient_descent(x, y, m, c, learning_rate, epochs)

# Plot the line of best fit after final epoch
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, m * x + c, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.legend()
plt.title('Linear Regression: Line of Best Fit')
plt.show()

# Predict office price when size is 100 sq. ft using the trained model
predicted_price = m * 100 + c
print(f"Predicted office price for 100 sq. ft: {predicted_price}")
