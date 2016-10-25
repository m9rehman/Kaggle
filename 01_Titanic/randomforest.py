import pandas as pd 
import sklearn
from sklearn.cross_validation import KFold
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import re

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

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
algRF = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

#Cross validation using 3 folds
kf = KFold(trainingSet.shape[0],n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(algRF, trainingSet[features], trainingSet["Survived"], cv=kf)
# print(scores.mean())


#Feature Engineering
# Counting the family members using SibSp and Parch
# How a rich a person was based on the length of their name using a lambda function with pandas .apply 
# New Series are added to the dataframe

trainingSet["FamilySize"] = trainingSet["SibSp"] + trainingSet["Parch"]
trainingSet["NameLength"] = trainingSet["Name"].apply(lambda x: len(x))


#Function to get titles using regex
def get_title(name):

	titleSearch = re.search(' ([A-Za-z]+)\.',name)

	if titleSearch:
		return titleSearch.group(1)
	return ""

#Applying our get_title function and printing title counts
titles = trainingSet["Name"].apply(get_title)
print(pd.value_counts(titles))


