# SGD-Classifier
# Developed by: KISHORE K
# RegisterNumber: 212223040101 
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary modules.
2. Load the Iris dataset.
3. Split the data into training and testing sets
4. Create and train the SGD Classifier
5. Make predictions
6. Evaluate the model

## Program:
```
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = clf.predict(sample)
print("Predicted species:", iris.target_names[prediction][0])

```

## Output:
![Screenshot 2025-04-10 144925](https://github.com/user-attachments/assets/227bfcab-90dd-4f39-8bee-40ddedc49ccf)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
