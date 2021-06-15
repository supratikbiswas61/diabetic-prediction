import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("diabetes.csv")

print (df.head())

#use required features
cdf = df[['BMI','BloodPressure','Glucose','Outcome']]

#Training Data and Predictor Variable
# Use all data for training (train-test-split not used)
x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]
regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))

'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''

