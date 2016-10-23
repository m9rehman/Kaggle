import pandas as pd 

trainingSet = pd.read_csv('train.csv')

print(trainingSet.head(5))
# print(trainingSet.describe())

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

# print(trainingSet["Embarked"])


#Using Linear Regression based on Age to predict
m = -2 
b = 20
trainingSet["LinPredictions"] = m*trainingSet["Age"] + b
trainingSet["LinPredictions"][trainingSet["LinPredictions"] < 0] = int(0)
trainingSet.loc[trainingSet["LinPredictions"] > 0] = int(1)

print(trainingSet["LinPredictions"])
