##Step 1 Loading the dataset ds and imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report

ds = pd.read_csv('titanic.csv')

##Step 2 Handling Missing Values using Numpy and Sklearn
#Using Numpy we get the median and fill in the missing data values in the given dataset
ds['Age']=ds['Age'].fillna(np.median(ds['Age'].dropna()))

#using the pandas module we fill in the missing elements with mode()

ds['Embarked'] = ds['Embarked'].fillna(ds['Embarked'].mode()[0])

#Imputing missing cabin values
ds['Cabin']=ds['Cabin'].fillna('Missing')

#Encoding step(using first letter of the cabin , eg : 'C234'
cabin_map={cabin:i+1 for i,cabin in enumerate("ABCDEFG")}

#Assinging 0 for missing cabins
cabin_map['Missing']=0

ds['Cabin']=ds['Cabin'].apply(lambda x: cabin_map.get(x[0],0))

#Mapping 'Sex' to binary values for compatiblity
ds['Sex']=ds['Sex'].map({'male':0,'female':1})

#Dropping the PassengerId column if exists
if 'PassengerId' in ds.columns:
    ds=ds.drop(columns=['PassengerId'])
assert 'Survived' in ds.columns,"Survived column missing from the Dataset"

#Defining the features and splitting the dataset
features=['Pclass','Sex','SibSp','Parch','Fare','Age'] 
x = ds[features]
y = ds['Cabin']

#Splitting it into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Training a random forest classifier with hyperparameter tuning
param_grid = {
    'n_estimators':[100,200,300],
    'max_depth':[None,10,20],
    'min_samples_split':[2,5,10],
    }
#using GridSearchCV
grid_search=GridSearchCV(RandomForestClassifier(random_state=123),param_grid,cv=3,scoring='accuracy')
grid_search.fit(x_train,y_train)
clf=grid_search.best_estimator_

#Evaluation of the model accuracy
y_pred=clf.predict(x_test)
print("Model Accuracy : ",accuracy_score(y_test,y_pred))
print("Classification Report :")
print(classification_report(y_test,y_pred,zero_division=0))
print('\n')

##Step 3 data exploration using pandas

print(ds.head())
print('\n')
print(ds.info())
print('\n')
print(ds.describe())
print('\n')
print(ds.isnull().sum())
print('\n')

##Step 4 data visualisation using matplotlib and seaborn

#plotting the dataset
ds.plot()
plt.title('Titanic Data Visualization')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()

# Plot Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(ds['Age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Visualize Fare Distribution by Passenger Class
plt.figure(figsize=(12, 8))
sns.boxplot(
    x='Pclass', 
    y='Fare', 
    data=ds,
    hue='Pclass',
    palette=['gold', 'silver', '#cd7f32']  # Reflect the ranking of classes
)
plt.title('Fare Distribution Across Passenger Classes', fontsize=16, fontweight='bold')
plt.xlabel('Passenger Class (1 = First, 2 = Second, 3 = Third)', fontsize=12)
plt.ylabel('Fare (in Dollars)', fontsize=12)
plt.xticks(
    ticks=[0, 1, 2], 
    labels=['First Class (Luxury)', 'Second Class (Comfort)', 'Third Class (Economy)'], 
    fontsize=10
)
medians = ds.groupby('Pclass')['Fare'].median()
for idx, median in enumerate(medians):
    plt.text(idx, median + 10, f'Median: ${median:.2f}', ha='center', fontsize=10, color='black')
plt.tight_layout()
plt.show()


# Plotting the survival rates by gender and passenger class
plt.figure(figsize=(10, 6))
sns.barplot(data=ds, x='Pclass', y='Survived', hue='Sex', palette='coolwarm')
plt.title('Survival Rates by Gender and Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.show()

# Display correlations between selected features and Survival
plt.figure(figsize=(10, 8))
selected_features = ['Survived', 'Pclass', 'Fare', 'Age', 'SibSp', 'Parch']
correlation_matrix = ds[selected_features].corr()
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    linewidths=0.5,  # Add gridlines for clarity
    cbar_kws={'label': 'Correlation Coefficient'}  # Add label for the color bar
)
plt.title('Correlation Between Selected Features and Survival', fontsize=16, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()


