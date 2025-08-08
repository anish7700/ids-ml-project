import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from src.load_data import load_nsl_kdd
from src.preprocess import prepare_data

def evaluate(model_path, train_path, test_path):
    model = joblib.load(model_path)
    _, df_test = load_nsl_kdd(train_path, test_path)
    X_test, y_test = prepare_data(df_test, binary=True)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds, labels=["normal","attack"])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=["normal","attack"], yticklabels=["normal","attack"])
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig("models/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate("models/ids_pipeline.joblib","data/KDDTrain+.txt","data/KDDTest+.txt")
