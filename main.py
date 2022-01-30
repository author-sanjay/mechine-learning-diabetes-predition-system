import numpy as np
import numpy as num
import pandas as pd
from sklearn.preprocessing import StandardScaler #standarize data
from sklearn.model_selection import train_test_split  #split data into training and test data
from sklearn import svm  #support avector machine
from sklearn.metrics import accuracy_score


# Data Collection and analysis

# loading diabetes dataset to a pandas dataframe

diabetes_dataset = pd.read_csv('D:\projects\ml projects\diabetes prediction\diabetes.csv')


# showing number of rows and columns in the dataset
#print(diabetes_dataset.shape)

#getting the statistical measure of data
#print(diabetes_dataset.describe())

#0 = non diabetic and 1= diabetic
#print(diabetes_dataset['Outcome'].value_counts())

#print(diabetes_dataset.groupby('Outcome').mean())





#seperating data and labels
x = diabetes_dataset.drop(columns= 'Outcome', axis=1)   #axis = 1 for column and 0 for dropping a row
y = diabetes_dataset['Outcome']
#print(x)
#print(y)






#DATA STANDARDIZATION
scaler = StandardScaler()
standardized_data = scaler.fit_transform(x)  # fitting inconsistant data into a comman range
# print(standardized_data)
x = standardized_data
y = diabetes_dataset['Outcome']
#print(x)
#print(y)





#spliting data into training and test data
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)     #0.2 is 20% of data as train data stratifyY means that we are spliting the dta in propotion to y else it may happen that all the diabetic cases may go in x train and all the non diabitic for test side.
print(x.shape, x_test.shape, x_train.shape)



#training the model

classifer= svm.SVC(kernel='linear')

# training SVM classifier
classifer.fit(x_train, y_train)

#model evaluation
    #accuracy Score
x_train_prediction = classifer.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)      #predictions are stored in atrainprediction and then we are compairiing the prediction with ytrain so that we get accuracy
print('Accuracy score of the training data:', training_data_accuracy)




#model evaluation
    #accuracy Score of test data
x_test_prediction = classifer.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)      #predictions are stored in atestprediction and then we are compairiing the prediction with ytest so that we get accuracy
print('Accuracy score of the test data:', test_data_accuracy)







#building a system for predictions
print("Enter No of Pregnancies")
Pregnancies = int(input())
print("Enter  Glucose Level")
Glucose = int(input())
print("Enter BloodPressure Level")
BloodPressure = int(input())
print("Enter SkinThickness")
SkinThickness = int(input())
print("Enter Insulin")
Insulin = int(input())
print("Enter BMI")
BMI = float(input())
print("Enter DiabetesPedigreeFunction")
DiabetesPedigreeFunction = float(input())
print("Enter Age")
Age = int(input())



input_data = (Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)


#changing data into numpy array because processing numpy array is easy

input_data_as_numpy_array= np.array(input_data)

#reshaping the array as we ara prdicting for one data only
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)


#standardizationd of inpyut data
std_data= scaler.transform(input_data_reshaped)
#print(std_data)


prediction = classifer.predict(std_data)


if(prediction[0]==0):
    print("Personn Is not diabatic")
else:
    print("Personn Is diabatic")