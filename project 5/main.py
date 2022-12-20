import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv(r'C:\Users\sakth\PycharmProjects\python-projects\palan_project\projects\project 5\payment_fraud.csv')
df.head()

# Split dataset up into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['label','paymentMethod'], axis=1), df['label'],
    test_size=0.33, random_state=17)

clf = LogisticRegression().fit(X_train, y_train)

# Make predictions on test set
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))

# Compare test set predictions with ground truth labels
print(confusion_matrix(y_test, y_pred))