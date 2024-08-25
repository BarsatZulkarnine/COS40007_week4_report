import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

train_data = pd.read_csv('cleaned_train_data.csv')

# Using only SP features generate a decisiontree mod
sp_columns = [col for col in train_data.columns if col.endswith('SP')]
X_sp = train_data[sp_columns]
y = train_data['Class']

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_sp, y)

# Print the tree using export_tex
tree_rules = export_text(decision_tree, feature_names=list(X_sp.columns))
print("Decision Tree Rules for SP Features:\n")
print(tree_rules)

