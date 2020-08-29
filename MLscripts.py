import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier 




# Import `Dense` from `keras.layers`
from keras.layers import Dense #this is used to define each layer of the neural network
from keras.models import Sequential #import the model we will be using

#Imports data frame
df = pd.read_csv("infile.csv")


def RFC(train_x, test_x, train_y, test_y):
    print("Random Forest Classifier")
    print(" ")
    clf = RandomForestClassifier()
    clf = clf.fit(train_x,train_y)
    y_pred = clf.predict(test_x)
    PrintResults(test_y, y_pred)
    
    temp = GetOutData(y_pred, test_y)
    ret = ["Random Forest Classifier"] + temp
    return ret
    
def GNB(train_x, test_x, train_y, test_y):
    print("Naive Bayes Classifier")
    print(" ")
    clf = GaussianNB()
    clf = clf.fit(train_x,train_y)
    y_pred = clf.predict(test_x)
    PrintResults(test_y, y_pred)
    
    temp = GetOutData(y_pred, test_y)
    ret = ["Naive Bayes Classifier"] + temp
    return ret
    
def SDG(train_x, test_x, train_y, test_y):
    print("Scholastic Gradient Descent Classifier")
    print(" ")
    clf = SGDClassifier()
    clf = clf.fit(train_x,train_y)
    y_pred = clf.predict(test_x)
    PrintResults(test_y, y_pred)
    
    temp = GetOutData(y_pred, test_y)
    ret = ["Scholastic Gradient Descent Classifier"] + temp
    return ret
    
def Kmean(train_x, test_x, train_y, test_y):
    print("Kmeans")
    clf = KMeans(n_clusters=2)
    clf.fit(train_x)
    y_pred = clf.predict(test_x)
    PrintResults(test_y, y_pred)
    
    temp = GetOutData(y_pred, test_y)
    ret = ["Kmeans"] + temp
    return ret
    
    
def Ncentroid(train_x, test_x, train_y, test_y):
    print("Nearest Centroid Classifier")
    print(" ")
    clf = NearestCentroid()
    clf = clf.fit(train_x,train_y)
    y_pred = clf.predict(test_x)
    PrintResults(test_y, y_pred)
    
    temp = GetOutData(y_pred, test_y)
    ret = ["Nearest Centroid Classifier"] + temp
    return ret 
    
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
    
    temp = GetOutData(outcome, test_y)
    ret = ["Artificial Neural Network"] + temp
    return ret 
    
    
def PrintResults(test_y, y_pred):
    print("Accuracy: {0:.2f}%".format(100*accuracy_score(y_pred, test_y)))
    print(" ")
    #Display the Confusion Matrix and Classification Report 
    print("Confusion Matrix:")
    print(confusion_matrix(y_pred, test_y))
    print(" ")
    print("Classification Report:")
    print(classification_report(y_pred, test_y))
    
def GetOutData(y_pred, test_y):
    precisionMi = precision_score(test_y, y_pred, average='micro')
    precisionMa = precision_score(test_y, y_pred, average='macro')
    precisionW = precision_score(test_y, y_pred, average='weighted')
    recallMi = recall_score(test_y, y_pred, average='micro')
    recallMa = recall_score(test_y, y_pred, average='macro')
    recallW = recall_score(test_y, y_pred, average='weighted')
    f1Mi = f1_score(test_y, y_pred, average='micro')
    f1Ma = f1_score(test_y, y_pred, average='macro')
    f1W = f1_score(test_y, y_pred, average='weighted')
    acc = accuracy_score(test_y, y_pred)
    
    ret = [acc,precisionW ,recallW, f1W, precisionMi, precisionMa, recallMi, recallMa, f1Mi, f1Ma]
    return ret
    
 

    
train_x, test_x, train_y, test_y = train_test_split(df.drop('benign', axis =1 ),df['benign'],test_size=0.3, random_state=42)

rfc = RFC(train_x, test_x, train_y, test_y)
gnb = GNB(train_x, test_x, train_y, test_y)
sdg = SDG(train_x, test_x, train_y, test_y)
kmean = Kmean(train_x, test_x, train_y, test_y)
ncent = Ncentroid(train_x, test_x, train_y, test_y)
ann = ANN(train_x, test_x, train_y, test_y)

OutData = [ann, ncent, kmean, sdg, gnb, rfc]

outdf = pd.DataFrame(OutData, columns = ['Model', 'Accuracy','Weighted Avg Precision','Weighted Avg Recall','Weighted Avg F-1 Score','Micro Avg Precision','Macro Avg Precision','Micro avg Recall','Macro avg Recall','Micro Avg F-1 Score','Macro Avg F-1 Score'])   

print(outdf.head())

outdf.to_csv('outfile.csv') 
