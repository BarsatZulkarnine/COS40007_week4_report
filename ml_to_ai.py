import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib

# Step 12: Load the 1000 unseen data points
test_data = pd.read_csv('test_data.csv')

# Step 13: Load the trained model
best_model = joblib.load('best_model.pkl')

# Step 14: Prepare the test set with the same features as the training set
# Load the training data to get the selected features used in training
train_data = pd.read_csv('cleaned_train_data.csv')
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']

selected_features = X_train.columns[:20]  

X_test = test_data[selected_features]  
y_test = test_data['Class']

# Step 15: Make predictions with the loaded model
y_pred_test = best_model.predict(X_test)

# Step 16: Measure the performance of the best model on the unseen data points
print("Performance of the best model on 1000 unseen data points:")
print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))

# Step 17: Measure the performance of other models on the unseen data points
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVC": SVC(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000)
}

X_train_selected = X_train[selected_features]

for name, model in models.items():
    model.fit(X_train_selected, y_train)  
    y_pred_other = model.predict(X_test)
    
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred_other))
    print(confusion_matrix(y_test, y_pred_other))

print("Model performance comparison completed.")
