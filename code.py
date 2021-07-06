import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics  

#Importing the dataset
data_set = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv", header = 0)

#Splitting the dataset
data_X= data_set.iloc[:, :-1]
data_y= data_set.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)

#Creating Linear Regression Model object
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)

percentage_y_pred = regr.predict(X_test)
regr.score(data_X,data_y)

#To display accuracy of the model developed
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, percentage_y_pred)) 

#Plotting a graph showing variation between original y values in dataset and predicted y values by the Linear Regression Model
line = regr.coef_*data_X+regr.intercept_
plt.scatter(data_X, data_y,  color='black')
plt.plot(data_X,line,color='blue', linewidth=3)
plt.show()

#Predicting percentage df student that studies for 9.25 hrs per day
hours = 9.25
hours = np.array(hours)
hours = hours.reshape(1,1)
percentage_y_pred = regr.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(percentage_y_pred[0]))
