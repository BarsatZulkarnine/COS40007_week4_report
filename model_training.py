import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE
import joblib

# 6) Does the training process need all features? If not, can you apply some feature selection
# technique to remove some features? Justify your reason of feature selection
train_data = pd.read_csv('cleaned_train_data.csv')
X = train_data.drop(columns=['Class'])
y = train_data['Class']
selector = RFE(DecisionTreeClassifier(random_state=42), n_features_to_select=20)
X_selected = selector.fit_transform(X, y)
print(f"Selected features: {X.columns[selector.support_]}")

# 7) Train multiple ML models (at least 5 including DecisionTreeClassifier) with your selected
# features.
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVC": SVC(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000)
}

# Initialize a list to store results
results = []

# 8) Evaluate each model with classification report and confusion matrix
for name, model in models.items():
    model.fit(X_selected, y)
    y_pred = model.predict(X_selected)
    report = classification_report(y, y_pred, output_dict=True)
    
    # Store the average weighted results in the results list
    results.append({
        'Model': name,
        'Accuracy': report['accuracy'],
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-Score': report['weighted avg']['f1-score']
    })
    
    print(f"Model: {name}")
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))

# 9) Compare all the models across different evaluation measures and generate a comparison table.
results_df = pd.DataFrame(results)
print("Comparison Table:")
print(results_df)

# 10) Now select your best performing model to use that as AI. Justify the reason of your selection
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_selected, y)

# 11) Now save your selected model
joblib.dump(best_model, 'best_model.pkl')



