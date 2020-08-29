import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier 


# Import `Dense` from `keras.layers`
from keras.layers import Dense #this is used to define each layer of the neural network
from keras.models import Sequential #import the model we will be using

#Imports data frame
df = pd.read_csv("BenignHome2dataAndDDoSDarpaData1.csv")

# Here remove or edit any NAN values

pd.set_option('display.max_rows', None)
#sets null protocalls to 0 
df = df.fillna({'ip.proto': 0})

def RFC(train_x, test_x, train_y, test_y):
    print("Random Forest Classifier")
    print(" ")
    clf = RandomForestClassifier()
    clf = clf.fit(train_x,train_y)
    y_pred = clf.predict(test_x)
    print("Accuracy: {0:.2f}%".format(100*accuracy_score(y_pred, test_y)))
    print(" ")
    #Display the Confusion Matrix and Classification Report 
    print("Confusion Matrix:")
    print(confusion_matrix(y_pred, test_y))
    print(" ")
    print("Classification Report:")
    print(classification_report(y_pred, test_y))
    
def GNB(train_x, test_x, train_y, test_y):
    print("Naive Bayes Classifier")
    print(" ")
    clf = GaussianNB()
    clf = clf.fit(train_x,train_y)
    y_pred = clf.predict(test_x)
    print("Accuracy: {0:.2f}%".format(100*accuracy_score(y_pred, test_y)))
    print(" ")
    #Display the Confusion Matrix and Classification Report 
    print("Confusion Matrix:")
    print(confusion_matrix(y_pred, test_y))
    print(" ")
    print("Classification Report:")
    print(classification_report(y_pred, test_y))
    
def SDG(train_x, test_x, train_y, test_y):
    print("Scholastic Gradient Descent Classifier")
    print(" ")
    clf = SGDClassifier()
    clf = clf.fit(train_x,train_y)
    y_pred = clf.predict(test_x)
    print("Accuracy: {0:.2f}%".format(100*accuracy_score(y_pred, test_y)))
    print(" ")
    #Display the Confusion Matrix and Classification Report 
    print("Confusion Matrix:")
    print(confusion_matrix(y_pred, test_y))
    print(" ")
    print("Classification Report:")
    print(classification_report(y_pred, test_y))
    
def Kmean(train_x, test_x, train_y, test_y):
    print("Kmeans")
    clf = KMeans(n_clusters=2)
    clf.fit(train_x)
    y_pred = clf.predict(test_x)
    print("Accuracy: {0:.2f}%".format(100*accuracy_score(y_pred, test_y)))
    print(" ")
    #Display the Confusion Matrix and Classification Report 
    print("Confusion Matrix:")
    print(confusion_matrix(y_pred, test_y))
    print(" ")
    print("Classification Report:")
    print(classification_report(y_pred, test_y))
    
    
def Ncentroid(train_x, test_x, train_y, test_y):
    print("Nearest Centroid Classifier")
    print(" ")
    clf = NearestCentroid()
    clf = clf.fit(train_x,train_y)
    y_pred = clf.predict(test_x)
    print("Accuracy: {0:.2f}%".format(100*accuracy_score(y_pred, test_y)))
    print(" ")
    #Display the Confusion Matrix and Classification Report 
    print("Confusion Matrix:")
    print(confusion_matrix(y_pred, test_y))
    print(" ")
    print("Classification Report:")
    print(classification_report(y_pred, test_y))
    
def ANN(train_x, test_x, train_y, test_y):
    print("Artificial Neural Network")
    model = Sequential() #define the model being used
    model.add(Dense(8, input_dim=8, activation='relu')) #input layer and first hidden layer
    model.add(Dense(5, activation='relu')) #second hidden layer
    model.add(Dense(1, activation='sigmoid')) # output layer 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=15, batch_size=10)
    loss, accuracy = model.evaluate(test_x, test_y)
    print('Accuracy: %.2f' % (accuracy*100))
    print(" ")
    outcome = model.predict_classes(test_x)
    matrix = confusion_matrix(test_y, outcome)
    print(matrix)
    print(classification_report(outcome, test_y))
    
train_x, test_x, train_y, test_y = train_test_split(df.drop('benign', axis =1 ),df['benign'],test_size=0.3, random_state=42)

RFC(train_x, test_x, train_y, test_y)
GNB(train_x, test_x, train_y, test_y)
SDG(train_x, test_x, train_y, test_y)
Kmean(train_x, test_x, train_y, test_y)
Ncentroid(train_x, test_x, train_y, test_y)
ANN(train_x, test_x, train_y, test_y)