import pandas as pd 
import sklearn
from sklearn.cross_validation import KFold
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import re
import operator
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

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

#--------------------------------------------------------------------------
# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
algRF = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

#Cross validation using 3 folds
kf = KFold(trainingSet.shape[0],n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(algRF, trainingSet[features], trainingSet["Survived"], cv=kf)
# print(scores.mean())

#--------------------------------------------------------------------------
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
# print(pd.value_counts(titles))

#Title mapping, some are mapped to same v since they are so rare
titleMap = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6,
 "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in titleMap.items():
	titles[titles == k] = v

# print(pd.value_counts(titles))

#Adding to our dataframe
trainingSet["Title"] = titles
# print(trainingSet.head())

#--------------------------------------------------------------------------
#Family ID Mapping

familyIDMap = {}

def getFamilyID(row):
	lastName = row["Name"].split(",")[0]
	#How familyID is defined eg: Johnson5
	familyID = "{0}{1}".format(lastName, row["FamilySize"])

	#Checking our familyIDMap
	if familyID not in familyIDMap:
		if len(familyIDMap) == 0:
			currID = 1
		else:
		#Otherwise find max of id and add 1
			currID = (max(familyIDMap.items(), key=operator.itemgetter(1))[1] + 1)

		familyIDMap[familyID] = currID
	return familyIDMap[familyID]


familyIDs = trainingSet.apply(getFamilyID,axis = 1)
#Compress families < 3 members into one mapping
familyIDs[trainingSet["FamilySize"] < 3] = -1 

# print(pd.value_counts(familyIDs))
trainingSet["FamilyID"] = familyIDs

#--------------------------------------------------------------------------
#Selecting only certain features

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyID"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(trainingSet[features], trainingSet["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(features)), scores)
plt.xticks(range(len(features)), features, rotation='vertical')
plt.show()

# Pick only the four best features.
features = ["Pclass", "Sex", "Fare", "Title"]


alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

kf = KFold(trainingSet.shape[0],n_folds=3,random_state=1)
scores = cross_validation.cross_val_score(alg,trainingSet[features],trainingSet["Survived"],cv=kf)
print(scores.mean())
#--------------------------------------------------------------------------



