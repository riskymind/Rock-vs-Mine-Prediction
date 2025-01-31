import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from sklearn.preprocessing import StandardScaler



pd.set_option('future.no_silent_downcasting', True)
sonar_data = pd.read_csv("sonar data.csv", header=None)

# Separate data and label
X = sonar_data.iloc[:, :-1]
y = sonar_data.iloc[:, -1].replace({"R":0, "M":1}).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model using Logistics Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate Model
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)

# Accuracy score on trained and test data
train_data_accuracy = accuracy_score(X_train_prediction, y_train)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)

print(f"Accuracy: {test_data_accuracy * 100:.2f}%")

# Deploy the model
# joblib.dump(model, "rock_mine_classifier.pkl")

# input_data = [0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032]
# # Changing the input data to a numpy array
# input_data_as_numpy = np.asarray(input_data)
# # Reshape the np array as we are predicting for one instance
# input_data_reshaped = input_data_as_numpy.reshape(1, -1)
# pred = model.predict(input_data_reshaped)

# if pred[0] == 'R':
#   print("The object is a Rock")
# elif pred[0] == "M":
#   print("The object is a Mine")
# else:
#   print("The object is a unknown")

