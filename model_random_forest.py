import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class RandomForestModel:
    def __init__(self):
        self.model_name = "RandomForest"
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

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
        return pipeline

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)

    def preprocess(self):
        X = self.data.drop('success_indicator', axis=1)
        y = self.data['success_indicator']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        self.pipeline.fit(self.X_train, self.y_train)

    def test(self):
        y_pred = self.pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print(classification_report(self.y_test, y_pred))

    def predict(self, X):
        return self.pipeline.predict(X)
