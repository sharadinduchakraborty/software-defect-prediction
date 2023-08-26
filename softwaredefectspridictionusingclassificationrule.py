import numpy as np
import pandas as pd
import matplotlib as mlt

import matplotlib.pyplot as plt # ploting , visualization
import seaborn as sns # ploting
from sklearn import model_selection #scikit learn
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils
import warnings
warnings.filterwarnings("ignore")
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import feature_selection
from sklearn.feature_extraction import DictVectorizer
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import chart_studio.plotly  as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import os



df1=pd.read_csv("C:/Users/shara/OneDrive/Documents/python/datasets/jm1.csv")

df1.info()

df1.head()

df1.columns
df1.dtypes
df1.shape

cleancolumn = []
for i in range(len(df1.columns)):
    cleancolumn.append(df1.columns[i].replace(' ','').lower())
df1.columns = cleancolumn

df1.isnull().sum()

df1.describe()

##Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df1, nGraphShown, nGraphPerRow):
    nunique = df1.nunique()
    df1 = df1[[col for col in df1 if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df1.shape
    columnNames = list(df1)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df1.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()



#Correlation matrix
def plotCorrelationMatrix(df1, graphWidth):
    df1 = df1.dropna('columns') # drop columns with NaN
    df1 = df1[[col for col in df1 if df1[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df1.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df1.shape[1]}) is less than 2')
        return
    corr = df1.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for dataset', fontsize=15)
    plt.show()





# Scatter and density plots
def plotScatterMatrix(df1, plotSize, textSize):
    df1 = df1.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df1 being singular
    df1 = df1.dropna('columns')
    df1 = df1[[col for col in df1 if df1[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df1)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df1 = df1[columnNames]
    ax = pd.plotting.scatter_matrix(df1, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df1.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

from mpl_toolkits.mplot3d import Axes3D

plotPerColumnDistribution(df1, 10, 5)

plotCorrelationMatrix(df1, 7)

plotScatterMatrix(df1, 20, 10)

defects_true_false = df1.groupby('defects')['b'].apply(lambda x: x.count()) #defect rates (true/false)
print('False : ' , defects_true_false[0])
print('True : ' , defects_true_false[1])

trace = go.Histogram(
    x = df1.defects,
    opacity = 0.75,
    name = "Defects",
    marker = dict(color = 'green'))

hist_data = [trace]
hist_layout = go.Layout(barmode='overlay',
                   title = 'Defects',
                   xaxis = dict(title = 'True - False'),
                   yaxis = dict(title = 'Frequency'),
)
fig = go.Figure(data = hist_data, layout = hist_layout)
iplot(fig)


f,ax = plt.subplots(figsize = (15, 15))
sns.heatmap(df1.corr(), annot = True, linewidths = .5, fmt = '.2f')
plt.show()

trace = go.Scatter(
    x = df1.v,
    y = df1.b,
    mode = "markers",
    name = "Volume - Bug",
    marker = dict(color = 'darkblue'),
    text = "Bug (b)")

scatter_data = [trace]
scatter_layout = dict(title = 'Volume - Bug',
              xaxis = dict(title = 'Volume', ticklen = 5),
              yaxis = dict(title = 'Bug' , ticklen = 5),
             )
fig = dict(data = scatter_data, layout = scatter_layout)
iplot(fig)

trace1 = go.Box(
    x = df1.uniq_op,
    name = 'Unique Operators',
    marker = dict(color = 'blue')
    )
box_data = [trace1]
iplot(box_data)

def evaluation_control(df1):    
    evaluation = (df1.n < 300) & (df1.v < 1000 ) & (df1.d < 50) & (df1.e < 500000) & (df1.t < 5000)
    df1['defects'] = pd.DataFrame(evaluation)
    df1['defects'] = ['true' if evaluation == True else 'false' for evaluation in df1.defects]

evaluation_control(df1)
df1

X =df1.drop(["defects"],axis=1)

X.head()

y = df1[["defects"]]
y.head()

scaler = MinMaxScaler()
scl_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
x_train.shape  , y_train.shape

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc_model.fit(x_train,y_train)
svc_pred = svc_model.predict(x_test)
svc_score = accuracy_score(svc_pred,y_test)*100
svc_score

from sklearn.naive_bayes import GaussianNB
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(x_train,y_train)
naive_bayes_pred = naive_bayes_model.predict(x_test)
naive_bayes_score = accuracy_score(naive_bayes_pred,y_test)*100
naive_bayes_score


from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

k_fold = KFold(len(df), n_folds=10, shuffle=True, random_state=0)

svc_cv_model = SVC()
svc_cv_score = cross_val_score(svc_cv_model,X,y,cv=k_fold,scoring = 'accuracy')*100
svc_cv_score
svc_cv_score.mean()


naive_bayes_cv_model = GaussianNB()
naive_bayes_cv_score = cross_val_score(naive_bayes_cv_model,X,y,cv=k_fold,scoring = 'accuracy')*100
naive_bayes_cv_score
naive_bayes_cv_score
naive_bayes_cv_model.fit(X,y)
naive_bayes_cv_pred = naive_bayes_cv_model.predict(X)
naive_bayes_cv_score = accuracy_score(naive_bayes_cv_pred,y)*100
naive_bayes_cv_score

from sklearn.tree import DecisionTreeClassifier
tree_cv_score = cross_val_score(tree_model,X,y,cv=k_fold,scoring = 'accuracy')*100
tree_cv_score
tree_cv_score.mean()
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_cv_score = cross_val_score(logistic_model,X,y,cv=k_fold,scoring = 'accuracy')*100

logistic_cv_score
logistic_cv_score.mean()

from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,26)
scores = []
for k in k_range :
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(x_train,y_train)
    pred = KNN.predict(x_test)
    scores.append(accuracy_score(pred,y_test)*100)
    
print(pd.DataFrame(scores))

plt.plot(k_range,scores)
plt.xlabel("K for KNN")
plt.ylabel("Testing scores")
plt.show()

k_range = range(1,26)
scores = []
for k in k_range :
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN_cv_score = cross_val_score(KNN,X,y,cv=k_fold,scoring = 'accuracy')*100
    cv_score = scores.append(KNN_cv_score)
    
print(pd.DataFrame(scores))

KNN_cv_score.mean()

plt.scatter(X_train, y_train, color = 'red')
modelin_tahmin_ettigi_y = model.predict(X_train)
plt.plot(X_train, modelin_tahmin_ettigi_y, color = 'black')
plt.title('Line of Code - Bug', size = 15)  
plt.xlabel('Line of Code')  
plt.ylabel('Bug')  
plt.show() 


from sklearn import metrics   
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))  
print('RootMeanSquaredError(RMSE):',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

