import pandas as  pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split
import joblib

#random seed
seed=42

#read original  dataset
iris_df=pd.read_csv('data/iris.csv')

#selecting  features and target data
x=iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y=iris_df['Species']
#splitting data into train and test
#70% train and 30% test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=seed,stratify= y)

#create an  instance of K neighbours classifier
clf=KNeighborsClassifier(n_neighbors=10)

#train  the model
clf.fit( x_train,y_train)

#predict on the  test set
y_pred= clf.predict(x_test)

#calculate  accuracy
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy}") #Accuracy: 0.9333333333333333

#save the model
joblib.dump(clf, "output_model/kn_model.sav")
















