import pandas as pd 
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

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
