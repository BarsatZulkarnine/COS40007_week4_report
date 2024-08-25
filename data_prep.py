import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

data = pd.read_csv('vegemite.csv')

# 1) First you need to shuffle the dataset
data = shuffle(data, random_state=2)

# Randomly take out 1000 data points (rows) such as way that each class in those 1000 samples has near equal distribution (e.g. at least 300 samples from each class
train_data, test_data = train_test_split(data, test_size=1000, stratify=data['Class'], random_state=2)


train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Does the dataset have any constant value column. If yes, then remove them
train_data = train_data.loc[:, (train_data != train_data.iloc[0]).any()]

# 2) Does the dataset have any column with few integer values? If yes, then convert them to categorial feature
for col in train_data.select_dtypes(include='int').columns:
    train_data[col] = train_data[col].astype('category')

class_counts = train_data['Class'].value_counts()
print("Class distribution:\n", class_counts)


#3) Does the class have a balanced distribution? If not then perform necessary undersampling and oversampling or adjust class weights.


from imblearn.over_sampling import SMOTE

X_train = train_data.drop(columns='Class')
y_train = train_data['Class']

smote = SMOTE(random_state=2)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

train_data_resampled = pd.concat([X_train_resampled, y_train_resampled], axis=1)
print("Class distribution after oversampling:\n", y_train_resampled.value_counts())


# Save the cleaned train dataset
train_data_resampled.to_csv('cleaned_train_data.csv', index=False)

print(f"Final number of features: {train_data_resampled.shape[1]}")
