import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/home/anastasios/Documents/Python/fruitsText.csv')
#print(df)
#sns.countplot(df['fruit_name'],label="Count")
#plt.show()


###############
# Visualisation
###############

#df.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9),
 #                                       title='Box Plot for each input variable')
#plt.show()

#df.drop('fruit_label', axis=1).hist(bins=30,figsize=(9,9))
#plt.suptitle("Histogram for each numeric input variable")
#plt.show()

#################
#MACHINE LEARNING
#################

X = df[['mass', 'width', 'height', 'color_score']]
y = df['fruit_label']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


## BUILD MODELS

#LOGISTIC REGRESSION
def LR():
     from sklearn.linear_model import LogisticRegression

     logreg = LogisticRegression()
     logreg.fit(X_train,y_train)

     print('Accuracy of Logistic regression classifier on training set: {:.2f}'
          .format(logreg.score(X_train, y_train)))
     print('Accuracy of Logistic regression classifier on test set: {:.2f}'
          .format(logreg.score(X_test, y_test)))

#DECISION TREE
def DT():
     from  sklearn.tree import DecisionTreeClassifier

     clf = DecisionTreeClassifier().fit(X_train,y_train)

     print('Accuracy of Decision Tree classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
     print('Accuracy of Decision Tree classifier on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))
#K-NEAREST NEIGHBOURS

def KNN():
     from sklearn.neighbors import KNeighborsClassifier
     knn = KNeighborsClassifier()
     knn.fit(X_train,y_train)
     print('Accuracy of K-NN classifier on training set: {:.2f}'
           .format(knn.score(X_train, y_train)))
     print('Accuracy of K-NN classifier on test set: {:.2f}'
           .format(knn.score(X_test, y_test)))
     from sklearn.metrics import classification_report
     from sklearn.metrics import confusion_matrix
     pred = knn.predict(X_test)
     print(confusion_matrix(y_test, pred))
     print(classification_report(y_test, pred))

#KNN()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
def plot_fruit_knn(X, y, n_neighbors, weights):
    X_mat = X[['height', 'width']].as_matrix()
    y_mat = y.as_matrix()
     # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF'])
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)clf.fit(X_mat, y_mat)

