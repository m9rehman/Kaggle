import pandas as pd 

trainingSet = pd.read_csv('train.csv')

print(trainingSet.head(5))
# print(trainingSet.describe())

#Imputing our dataframe
trainingSet["Age"]=trainingSet["Age"].fillna(trainingSet["Age"].median())
# print(trainingSet.describe())

#Assigning a value to sex column
trainingSet["Sex"][trainingSet["Sex"] == 'male'] = 0
trainingSet.loc[trainingSet["Sex"] == 'female', "Sex"] = 1

print(trainingSet["Sex"])