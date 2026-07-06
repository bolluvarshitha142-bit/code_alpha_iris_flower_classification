import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Iris.csv")

# Remove Id column if present
if "Id" in data.columns:
    data = data.drop("Id", axis=1)

# Features and target
X = data.drop("Species", axis=1)
y = data["Species"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Sample prediction
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(sample)
print("Predicted Species:", prediction[0])
# Visualize the dataset
plt.figure(figsize=(8,5))
plt.scatter(data["SepalLengthCm"], data["PetalLengthCm"], c=pd.factorize(data["Species"])[0])
plt.title("Iris Flower Classification")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()