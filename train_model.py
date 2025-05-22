import pandas as pd
import dask.dataframe as dd
import pickle
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dask_ml.model_selection import GridSearchCV

# Load datasets (replace paths with your actual dataset locations)
df_imbalanced_pandas = pd.read_excel('final_dataset.xlsx')  # Imbalanced dataset
df_normal_pandas = pd.read_excel('final_normal.xlsx')  # Normal dataset
df_failure2_pandas = pd.read_excel('final_failure2.xlsx')  # Horizontal misalignment
df_failure3_pandas = pd.read_excel('final_failure3.xlsx')  # Vertical misalignment

# Convert to Dask DataFrames for distributed processing
df_imbalanced = dd.from_pandas(df_imbalanced_pandas, npartitions=10)
df_normal = dd.from_pandas(df_normal_pandas, npartitions=10)
df_failure2 = dd.from_pandas(df_failure2_pandas, npartitions=10)
df_failure3 = dd.from_pandas(df_failure3_pandas, npartitions=10)

# Sample 10% of each dataset to reduce size
df_imbalanced = df_imbalanced.sample(frac=0.1, random_state=42)
df_normal = df_normal.sample(frac=0.1, random_state=42)
df_failure2 = df_failure2.sample(frac=0.1, random_state=42)
df_failure3 = df_failure3.sample(frac=0.1, random_state=42)

# Assign labels
df_imbalanced['label'] = 1  # Imbalance
df_normal['label'] = 0  # Normal
df_failure2['label'] = 2  # Horizontal misalignment
df_failure3['label'] = 3  # Vertical misalignment

# Concatenate datasets
df_combined = dd.concat([df_normal, df_imbalanced, df_failure2, df_failure3])

# Remove any rows with index 0 (if applicable)
df_combined = df_combined.loc[df_combined.index != 0]

# Split into features (X) and labels (y)
X = df_combined.drop('label', axis=1)
y = df_combined['label']

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Apply standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize RandomForestClassifier
clf = RandomForestClassifier(random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20]
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_clf = grid_search.best_estimator_

# Make predictions on test set
y_pred = best_clf.predict(X_test_scaled)

# Convert Dask arrays to NumPy for accuracy calculation
y_test_np = y_test.compute()
y_pred_np = y_pred.compute()

# Calculate and print accuracy
accuracy = accuracy_score(y_test_np, y_pred_np)
print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)

# Save the trained model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_clf, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Load and verify the saved model (optional)
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)