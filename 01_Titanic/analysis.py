import pandas as pd 

trainingSet = pd.read_csv('train.csv')

print(trainingSet.head(5))
# print(trainingSet.describe())

#Imputing our dataframe
trainingSet["Age"]=trainingSet["Age"].fillna(trainingSet["Age"].median())
print(trainingSet.describe())