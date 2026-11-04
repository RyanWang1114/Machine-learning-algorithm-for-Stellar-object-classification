"""Compare four classifiers on a held-out subset of the SDSS17 catalog."""

import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from load_catalog import CLASS_NAMES, load_catalog
from prepare_features import prepare_features


def evaluate(path):
    features, target = load_catalog(path)
    train, test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    if y_train.nunique() < 2:
        raise ValueError("The training split needs at least two distinct classes.")
    train, test = prepare_features(train, test)

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="linear", random_state=42),
        "Naive Bayes": GaussianNB(),
    }
    class_ids = list(range(len(CLASS_NAMES)))
    print(f"Training rows: {len(train):,} | Test rows: {len(test):,}", flush=True)
    for name, classifier in classifiers.items():
        print(f"\n{name}", flush=True)
        classifier.fit(train, y_train)
        predictions = classifier.predict(test)
        report = classification_report(
            y_test, predictions, labels=class_ids, target_names=CLASS_NAMES, zero_division=0
        )
        print(report)
        print("Confusion matrix (rows = actual, columns = predicted; GALAXY, STAR, QSO):")
        print(confusion_matrix(y_test, predictions, labels=class_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path, help="Path to the original star_classification.csv")
    args = parser.parse_args()
    try:
        evaluate(args.catalog)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
