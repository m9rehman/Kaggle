import pandas as pd 
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np
from sklearn import cross_validation

trainingSet = pd.read_csv('train.csv')

#Columns used to make our predictions
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#Imputing our dataframe
trainingSet["Age"]=trainingSet["Age"].fillna(trainingSet["Age"].median())


#Assigning a value to sex column
trainingSet["Sex"][trainingSet["Sex"] == 'male'] = 0
trainingSet.loc[trainingSet["Sex"] == 'female', "Sex"] = 1

#Assign values to Embarked column
# na entries will be encodeded as S [most common embarking port]
# S(0), C(1) and Q(2)
trainingSet["Embarked"] = trainingSet["Embarked"].fillna('S')
trainingSet.loc[trainingSet["Embarked"] == 'S', "Embarked"] = 0
trainingSet["Embarked"][trainingSet["Embarked"] == 'C'] = 1
trainingSet["Embarked"][trainingSet["Embarked"] == 'Q'] = 2


#-------------------------------------------------------------------------------------
#Using Linear Regression 
algorithm = LinearRegression()

#Splitting our data 
#Returns the row indices for the training folds and the test folds
kf = KFold(trainingSet.shape[0], n_folds=3, random_state=1)

predictions = []
for trainIndices,testIndices in kf:
	#Subset of the dataframe with only training indices and features
	trainPredictors = trainingSet[features].iloc[trainIndices,:]
	#Setting our train training target 
	trainingTarget = trainingSet["Survived"].iloc[trainIndices]
	#Fitting our algorithm
algorithm.fit(trainPredictors,trainingTarget)

	#Testing on our test fold
testPredictions = algorithm.predict(trainingSet[features].iloc[testIndices,:])
predictions.append(testPredictions)
print(predictions)


#Doing some error analysis based on Kaggle's error metric (# of right predictions/ # of passengers)

#Concatenating the 3 predictions np arrays
predictions = np.concatenate(predictions,axis = 0)

#Mapping our predictions to a binary result
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

# trainingSet["Matches"] = predictions == trainingSet["Survived"]
# accuracy = (trainingSet["Matches"][trainingSet["Matches"]==True].value_counts()) / 891

#-------------------------------------------------------------------------------------
#Using Logistic Regression
algLogit = LogisticRegression(random_state=1)
#Calculates accuracy of each of the cross validation folds
scores = cross_validation.cross_val_score(algLogit, trainingSet[features], trainingSet["Survived"], cv=3)
print(scores.mean())

#-------------------------------------------------------------------------------------
#Preparing the testing set

testSet = pandas.read_csv("testSet.csv")

testSet["Age"] = testSet["Age"].fillna(trainingSet["Age"].median())

testSet.loc[testSet["Sex"]== 'male',"Sex"] = 0
testSet["Sex"][testSet["Sex"] == 'female'] = 1

testSet["Embarked"] = testSet["Embarked"].fillna('S')

testSet["Embarked"][testSet["Embarked"] == 'S'] = 0
testSet["Embarked"][testSet["Embarked"] == 'C'] = 1
testSet["Embarked"][testSet["Embarked"] == 'Q'] = 2

testSet["Fare"] = testSet["Fare"].fillna(testSet["Fare"].median())



