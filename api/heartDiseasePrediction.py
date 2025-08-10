import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("heart_disease_data.csv")


print(dataset.isnull())      #checking if there is any null values
print(dataset.shape)
x = dataset.iloc[:,:-1]      #independent columns or input
y = dataset['target']        #Dependent column or output

from sklearn.model_selection import train_test_split

#Splitting data in training (80%) and testing (20%)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

features = dataset.iloc[:,:-1]
print("Model trained Successfully")

y_prediction = model.predict(x_test)

comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_prediction
})

pd.set_option('display.max_rows', None)
print(comparison_df)

plt.figure(figsize=(8,7))
plt.plot(y_test.values, label='Actual value', marker='o')
plt.plot(y_prediction, label='Prediction', marker='p')
plt.title('Actual and Predicited heart disease')
plt.show()

from sklearn.metrics import confusion_matrix, classification_report

matrix = confusion_matrix(y_test, y_prediction)
print(matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_prediction, target_names=["No Heart Disease", "Heart Disease"]))

import joblib
joblib.dump(model, 'heart_disease_model.pkl')
