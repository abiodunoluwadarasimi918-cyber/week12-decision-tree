import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# Load the Iris dataset
df = pd.read_csv("/storage/emulated/0/Download/Iris.csv")

# Features and target
X = df[["SepalLengthCm", "SepalWidthCm",
        "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Split into training and test sets (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build and train the decision tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")

print("Accuracy :", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Plots: confusion matrix and the decision tree
fig, axes = plt.subplots(2, 1, figsize=(10, 14))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, ax=axes[0], cmap="Blues"
)
axes[0].set_title("Confusion Matrix")

plot_tree(model, feature_names=X.columns,
          class_names=model.classes_, filled=True, ax=axes[1])
axes[1].set_title("Decision Tree")

plt.tight_layout()
plt.savefig("decision_tree_results.png")
plt.show()