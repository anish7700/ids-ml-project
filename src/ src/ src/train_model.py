import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.load_data import load_nsl_kdd
from src.preprocess import prepare_data, make_preprocessor
from sklearn.pipeline import Pipeline

def train_and_save(train_path, test_path, save_path="models/ids_pipeline.joblib"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_train, df_test = load_nsl_kdd(train_path, test_path)
    X_train, y_train = prepare_data(df_train, binary=True)
    X_test, y_test = prepare_data(df_test, binary=True)

    pre = make_preprocessor(X_train)

    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced")

    pipeline = Pipeline([
        ("preprocessor", pre),
        ("clf", clf)
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(pipeline, save_path)
    print(f"Saved pipeline to {save_path}")

if __name__ == "__main__":
    train_path = "data/KDDTrain+.txt"
    test_path = "data/KDDTest+.txt"
    train_and_save(train_path, test_path)
