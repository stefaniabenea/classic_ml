from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import evaluate_model, accuracies
from models import models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# data load and scale
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
df["target"]= data.target
X=df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


for name, model in models.items():
   evaluate_model(model,X_train_scaled,y_train,X_test_scaled,y_test, name)

results_df = pd.DataFrame([
    {"Model": name, "Accuracy": acc}
    for name, acc in accuracies.items()
])
results_df.to_csv("results.csv", index=False)

plt.figure(figsize=(10,5))
plt.bar(accuracies.keys(),accuracies.values())
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()