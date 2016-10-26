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
#Ensembling multiple algorithms
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# The algorithms we want to ensemble.
# We're using the more linear features for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(trainingSet.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = trainingSet["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, features in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(trainingSet[features].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(trainingSet[features].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == trainingSet["Survived"]]) / len(predictions)
print(accuracy)

#--------------------------------------------------------------------------
#Applying our changes to the test set
# 
#Read the test set into titanic_test
titles = titanic_test["Name"].apply(get_title)
# We're adding the Dona title to the mapping, because it's in the test set, but not the training set
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles
# Check the counts of each unique title.
print(pd.value_counts(titanic_test["Title"]))

# Now, we add the family size column.
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

# Now we can add family ids.
# We'll use the same ids that we did earlier.
print(family_id_mapping)

family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))
