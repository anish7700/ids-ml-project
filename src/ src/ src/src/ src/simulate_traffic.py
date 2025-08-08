import joblib
from src.load_data import load_nsl_kdd
from src.preprocess import prepare_data

def simulate_and_predict(model_path):
    model = joblib.load(model_path)
    _, df_test = load_nsl_kdd("data/KDDTrain+.txt","data/KDDTest+.txt")
    X_test, y_test = prepare_data(df_test, binary=True)
    sample = X_test.sample(10, random_state=42)
    preds = model.predict(sample)
    out = sample.copy()
    out['predicted'] = preds
    print(out[['protocol_type','service','flag','predicted']])

if __name__ == "__main__":
    simulate_and_predict("models/ids_pipeline.joblib")
