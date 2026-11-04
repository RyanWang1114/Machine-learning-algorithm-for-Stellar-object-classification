import matplotlib.pyplot as plt
import pandas as pd

# Data for each classifier
classifiers = ["Random Forest", "Logistic Regression", "SVM", "Naive Bayes"]

# Precision, Recall, F1-Score for each class and classifier
data = {
    "Precision": {
        0: [0.98, 0.83, 0.83, 0.85],
        1: [0.98, 0.79, 0.85, 0.42],
        2: [0.97, 0.91, 0.95, 0.63]
    },
    "Recall": {
        0: [0.99, 0.94, 0.97, 0.66],
        1: [0.99, 0.55, 0.53, 0.53],
        2: [0.93, 0.85, 0.85, 0.88]
    },
    "F1-Score": {
        0: [0.98, 0.89, 0.90, 0.74],
        1: [0.99, 0.64, 0.66, 0.47],
        2: [0.95, 0.88, 0.90, 0.73]
    },
    "Accuracy": [0.98, 0.84, 0.85, 0.67]
}

# Plotting Precision by Class
precision_data = pd.DataFrame(data["Precision"], index=classifiers)
precision_data.plot(kind="bar", figsize=(10, 6), color=['#5b9bd5', '#ed7d31', '#a5a5a5'])
plt.title("Precision Comparison by Class")
plt.xlabel("Classifier")
plt.ylabel("Precision")
plt.legend(title="Class")
plt.show()

# Plotting Recall by Class
recall_data = pd.DataFrame(data["Recall"], index=classifiers)
recall_data.plot(kind="bar", figsize=(10, 6), color=['#5b9bd5', '#ed7d31', '#a5a5a5'])
plt.title("Recall Comparison by Class")
plt.xlabel("Classifier")
plt.ylabel("Recall")
plt.legend(title="Class")
plt.show()

# Plotting F1-Score by Class
f1_score_data = pd.DataFrame(data["F1-Score"], index=classifiers)
f1_score_data.plot(kind="bar", figsize=(10, 6), color=['#5b9bd5', '#ed7d31', '#a5a5a5'])
plt.title("F1-Score Comparison by Class")
plt.xlabel("Classifier")
plt.ylabel("F1-Score")
plt.legend(title="Class")
plt.show()

# Plotting Overall Accuracy
plt.figure(figsize=(10, 6))
plt.bar(classifiers, data["Accuracy"], color='#70ad47')
plt.title("Overall Accuracy Comparison")
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()
