import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["protocol_type","service","flag"]

def prepare_data(df, binary=True):
    df = df.copy()
    if binary:
        df['label'] = df['label'].apply(lambda x: 'normal' if x.strip() == 'normal' else 'attack')
    X = df.drop(columns=['label'])
    y = df['label'].values
    return X, y

def make_preprocessor(X):
    numeric_features = [c for c in X.columns if c not in CATEGORICAL]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), CATEGORICAL),
        ],
        remainder='drop'
    )
    return preprocessor
