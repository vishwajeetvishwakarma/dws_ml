import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf

class ANNModel:
    def __init__(self):
        self.model_name = "ANN"
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        numeric_features = ['stars']
        categorical_features = ['category', 'main_promotion', 'color']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        return pipeline

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)

    def preprocess(self):
        X = self.data.drop('success_indicator', axis=1)
        y = self.data['success_indicator']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        self.pipeline.fit(self.X_train, self.y_train, model__epochs=10)  # Adjust epochs as needed

    def test(self):
        y_pred = (self
