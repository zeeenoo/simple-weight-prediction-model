import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #to show data in the screen
import seaborn as sns #used to 
from sklearn.model_selection import train_test_split #where the model exists
from sklearn.linear_model import LinearRegression #to use the linear regression



#import data from csv file
df = pd.read_csv('weight-height.csv')
df.head() #to display the first 5 rows of the data

#analyze the data
df.describe() #to display the statistical data of the data
df.corr() #to check data correlation

#prepare data for training
x = df.iloc[:,:-1].values #we give him the height
y = df.iloc[:,1].values #the weight that we predict

#split data into training and test sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)  #to split the data into 2 parts (training and test) 0.2 means 20% of the data are for test 

#train the model
regressor = LinearRegression() #to create the model
regressor.fit(x_train,y_train) #to train the model on the x and y 


#test the model
y_pred = regressor.predict(x_test) #to predict the weight of the test data

#compare the actual output values for x_test with the predicted values
df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred}) 

#evaluate the model

h=input('enter the height')
pred = regressor.predict(np.array([[h]]))
print(f'the predicted weight for the height {h}m is {pred[0]:.2f} kg')

#check model accuracy
print('Testing score accuracy: ',regressor.score(x_test,y_test))


#plot the regression line
plt.scatter(x_train,y_train,color='red') # training data visualised in dots
plt.plot(x_train,regressor.predict(x_train),color='blue') # regression line
plt.title('Weight vs Height (Training set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()








