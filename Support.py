from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

def showErrors(Initial,AfterPred,Correct,RegressionName,StudyObject,show=True):
    if show == True:
        print(str(RegressionName)+' Regression '+str(StudyObject)+' Prediction')
        print('     After Prediction  / Initial')
        print('MAE:',mean_absolute_error(AfterPred,Correct),'/',mean_absolute_error(Initial,Correct))
        print('MSE:',mean_squared_error(AfterPred,Correct),'/',mean_squared_error(Initial,Correct))
        #print('r_2:', r2_score(AfterPred, Correct),'/',r2_score(Initial,Correct))
    return mean_absolute_error(AfterPred,Correct),mean_squared_error(AfterPred,Correct)

def bestLambda(df,regressionName,returnError=0):
    lambdaDict = {}
    for i in np.linspace(0,5,100):
        X = df[['SPO2_Apel', 'x', 'y', 'z', 'xSum', 'ySum', 'zSum', 'Array']]
        y = df[['spo2_Berry']].values
        lasso = Lasso(alpha=i)
        lasso.fit(X,y)
        y_pred = lasso.predict(df[['SPO2_Apel', 'x', 'y', 'z', 'xSum', 'ySum', 'zSum', 'Array']])
        df['SPO2_pred'] = y_pred
        lambdaDict.update({i:showErrors(df['SPO2_Apel'],df['SPO2_pred'].values,df['spo2_Berry'].values,regressionName,'SPO2',show=False)[returnError]})
    key_list = list(lambdaDict.keys()) 
    val_list = list(lambdaDict.values())
    print('Best lambda for',regressionName,'is',key_list[val_list.index(min(val_list))])
    return key_list[val_list.index(min(val_list))]


def plotSPO2(regressionName,df2predict,graphsFolderPath):
    plt.figure()
    plt.subplot(211)
    plotTitle = str(regressionName)+' Regression SPO2'
    plt.title(plotTitle)
    plt.plot(df2predict['TotalSec'],df2predict['SPO2_Apel'], label= 'Medical Device')
    plt.plot(df2predict['TotalSec'],df2predict['spo2_Berry'], label = 'Apel Actual')
    plt.plot(df2predict['TotalSec'],df2predict['SPO2_pred'], label = 'Apel Predicted')
    plt.legend()
    plt.subplot(212)
    plt.plot(df2predict['TotalSec'],df2predict['xPer'],'magenta', label= 'x')
    plt.plot(df2predict['TotalSec'],df2predict['yPer'], 'teal' ,label = 'y')
    plt.plot(df2predict['TotalSec'],df2predict['zPer'],'limegreen' ,label = 'z')
    plt.plot(df2predict['TotalSec'],df2predict['TotalChange'],'limegreen' ,label = 'Sum')     
    plt.legend()
    plt.savefig(graphsFolderPath+'/'+plotTitle)
    plt.show()

def plotHR(regressionName,df2predict,graphsFolderPath):
    plt.figure()
    plt.subplot(211)
    plotTitle = str(regressionName)+' Regression HR'
    plt.title(plotTitle)
    plt.plot(df2predict['TotalSec'],df2predict['pr_Berry'], label= 'Medical Device')
    plt.plot(df2predict['TotalSec'],df2predict['HR_Apel'], label = 'Apel Actual')
    plt.plot(df2predict['TotalSec'],df2predict['Pr_pred'], label = 'Apel Predicted')
    plt.legend()
    plt.subplot(212)
    plt.plot(df2predict['TotalSec'],df2predict['xPer'],'magenta', label= 'x')
    plt.plot(df2predict['TotalSec'],df2predict['yPer'], 'teal' ,label = 'y')
    plt.plot(df2predict['TotalSec'],df2predict['zPer'],'limegreen' ,label = 'z')  
    plt.plot(df2predict['TotalSec'],df2predict['TotalChange'],'limegreen' ,label = 'Sum')          
    plt.legend()
    plt.savefig(graphsFolderPath+'/'+plotTitle)
    plt.show()