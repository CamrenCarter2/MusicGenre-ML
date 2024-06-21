import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import joblib

# 1. Load and Concatenate Data Efficiently
data_files = [os.path.join('Data', filename) for filename in os.listdir('Data') if filename.endswith('.csv')]
data = pd.concat((pd.read_csv(f) for f in data_files), ignore_index=True)

# 2. Drop 'filename' and Separate Features and Labels in One Step
X = data.drop(columns=['filename', 'label'])
y = data['label']

valid_rows = y.notna()
X = X[valid_rows]
y = y[valid_rows]

# 3. Create a Preprocessing Pipeline for Imputation and Scaling
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 4. Split Data Before Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Fit and Transform the Pipeline on Training Data, Transform Test Data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# 6. Train Model (Consider Hyperparameter Tuning)
model = KNeighborsClassifier(n_neighbors=5)  # You could explore different values of n_neighbors
model.fit(X_train_processed, y_train)
joblib.dump(model, 'genre_classifier_model.pkl')

# 7. Evaluate Model
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')