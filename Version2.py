import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import Support
from sklearn.preprocessing import StandardScaler
import datetime
import os
import warnings
warnings.simplefilter("ignore")



# Give the paths for TRAINING DATA
pathApeljson = '/home/anastasios/Documents/ElectricalEngeneeringMasterDegree/MedicalProject/Data/apel707.json'
pathBerryCSV = '/home/anastasios/Documents/ElectricalEngeneeringMasterDegree/MedicalProject/Data/berry707.csv'

#give the path for TEST DATA

PathTestApel = '/home/anastasios/Documents/ElectricalEngeneeringMasterDegree/MedicalProject/Data/apel1307.json'
PathTestBerry = '/home/anastasios/Documents/ElectricalEngeneeringMasterDegree/MedicalProject/Data/berry1307.csv'

# Create folder to save the graphs for each day
today = str(datetime.date.today())
files = [file for file in os.listdir('/home/anastasios/Documents/ElectricalEngeneeringMasterDegree/MedicalProject/Graps')]
graphsFolderPath = '/home/anastasios/Documents/ElectricalEngeneeringMasterDegree/MedicalProject/Graps/'+today
if today not in files:
    try:
        os.mkdir(graphsFolderPath)
    except OSError:
        print ("Creation of the directory %s failed" % graphsFolderPath)
    else:
        print ("Successfully created the directory %s " % graphsFolderPath)

# Begin with the data of apel device
dfApel = pd.read_json(pathApeljson).transpose()
dfApel =dfApel[[ 'timestamp','HR', 'SPO2',  'xdata', 'ydata', 'zdata']]  # Remove useless columns

dfApel = dfApel.dropna(subset=['timestamp']) #clear the data
def get_time(time):  # function to split the timstamp column
    try:
        a= str(time).split(' ')[1].split(':')
    except:
        a = ['nan','nan','nan']
    if int(a[0]) > 12:
        a[0] = int(a[0]) - 12
    elif int(a[0]) == 0:
        a[0] = 12
    return a
dfApel['Hour'] = dfApel['timestamp'].apply(lambda x: int(get_time(x)[0]))
dfApel['Minutes'] = dfApel['timestamp'].apply(lambda x: int(get_time(x)[1]))
dfApel['Seconds'] = dfApel['timestamp'].apply(lambda x: int(get_time(x)[2]))

def getMove(move): # function which return the average of accelarator's values
    a = str(move).split(';')
    a.pop(-1)
    a = list(map(float, a))
    return sum(a)/len(a)
def getSumMove(move): # function which return the sum of accelarator's values
    a = str(move).split(';')
    a.pop(-1)
    a = list(map(float, a))
    return sum(a)
dfApel['x'] = dfApel['xdata'].apply(lambda x:getMove(x))
dfApel['y'] = dfApel['ydata'].apply(lambda x:getMove(x))
dfApel['z'] = dfApel['zdata'].apply(lambda x:getMove(x))
dfApel["Array"] = ((dfApel['x']**2)+(dfApel['y']**2)+(dfApel['z']**2))**1/2
dfApel['xSum'] = dfApel['xdata'].apply(lambda x:getSumMove(x))
dfApel['ySum'] = dfApel['ydata'].apply(lambda x:getSumMove(x))
dfApel['zSum'] = dfApel['zdata'].apply(lambda x:getSumMove(x))
dfApel = dfApel[[ 'HR', 'SPO2', 'Hour', 'Minutes', 'Seconds', 'x', 'y', 'z',  'xSum', 'ySum', 'zSum','Array']]
dfApel.columns = [ 'HR_Apel', 'SPO2_Apel', 'Hour', 'Minutes', 'Seconds', 'x', 'y', 'z','xSum', 'ySum', 'zSum', 'Array']

dfBerry = pd.read_csv(pathBerryCSV)

def get_time_berry(time):
    try:
        a= str(time).split(' ')[2].split(':')
    except:
        a = ['nan','nan','nan']
    return a
dfBerry['Hour'] = dfBerry['time'].apply(lambda x: int(get_time_berry(x)[0]))
dfBerry['Minutes'] = dfBerry['time'].apply(lambda x: int(get_time_berry(x)[1]))
dfBerry['Seconds'] = dfBerry['time'].apply(lambda x: int(get_time_berry(x)[2]))
dfBerry = dfBerry[['Hour', 'Minutes', 'Seconds', 'spo2', 'pr']]
dfBerry.columns = ['Hour', 'Minutes', 'Seconds', 'spo2_Berry', 'pr_Berry']
df = pd.merge(dfBerry,dfApel,on=['Hour', 'Minutes', 'Seconds'])
df = df.sort_values(by=['Hour', 'Minutes', 'Seconds'])
df['TotalSec'] = df['Hour']*60*60 + df['Minutes']*60 + df['Seconds']
df['x'] = df['x'] + 130
df['y'] = df['y'] + 50
df['z'] = df['z'] + 850
df['xPer'] = df[['x']].pct_change()
df['yPer'] = df[['y']].pct_change()
df['zPer'] = df[['z']].pct_change()
df = df.dropna()
df['TotalChange'] = df['xPer'] + df['yPer'] + df['zPer'] 


def insertNewData(NewApelDataPath,NewBerryDataPath):
        # Begin with the data of apel device
    dfNewApel = pd.read_json(NewApelDataPath).transpose()
    dfNewApel =dfNewApel[[ 'timestamp','HR', 'SPO2',  'xdata', 'ydata', 'zdata']]  # Remove useless columns

    dfNewApel = dfNewApel.dropna(subset=['timestamp', 'xdata', 'ydata', 'zdata']) #clear the data
    def get_time(time):  # function to split the timstamp column
        try:
            a= str(time).split(' ')[1].split(':')
        except:
            a = ['nan','nan','nan']
        if int(a[0]) > 12:
            a[0] = int(a[0]) - 12
        elif int(a[0]) == 0:
            a[0] = 12
        return a
    dfNewApel['Hour'] = dfNewApel['timestamp'].apply(lambda x: int(get_time(x)[0]))
    dfNewApel['Minutes'] = dfNewApel['timestamp'].apply(lambda x: int(get_time(x)[1]))
    dfNewApel['Seconds'] = dfNewApel['timestamp'].apply(lambda x: int(get_time(x)[2]))
    dfNewApel['Hour'] = dfNewApel['Hour'] 
    
    
    def getMove(move): # function which return the average of accelarator's values
        a = str(move).split(';')
        a.pop(-1)
        a = list(map(float, a))
        return sum(a)/len(a)
    def getSumMove(move): # function which return the sum of accelarator's values
        a = str(move).split(';')
        a.pop(-1)
        a = list(map(float, a))
        return sum(a)
    dfNewApel['x'] = dfNewApel['xdata'].apply(lambda x:getMove(x))
    dfNewApel['y'] = dfNewApel['ydata'].apply(lambda x:getMove(x))
    dfNewApel['z'] = dfNewApel['zdata'].apply(lambda x:getMove(x))
    dfNewApel["Array"] = ((dfNewApel['x']**2)+(dfNewApel['y']**2)+(dfNewApel['z']**2))**1/2
    dfNewApel['xSum'] = dfNewApel['xdata'].apply(lambda x:getSumMove(x))
    dfNewApel['ySum'] = dfNewApel['ydata'].apply(lambda x:getSumMove(x))
    dfNewApel['zSum'] = dfNewApel['zdata'].apply(lambda x:getSumMove(x))
    dfNewApel = dfNewApel[[ 'HR', 'SPO2', 'Hour', 'Minutes', 'Seconds', 'x', 'y', 'z',  'xSum', 'ySum', 'zSum','Array']]
    dfNewApel.columns = [ 'HR_Apel', 'SPO2_Apel', 'Hour', 'Minutes', 'Seconds', 'x', 'y', 'z','xSum', 'ySum', 'zSum', 'Array']

    dfNewBerry = pd.read_csv(NewBerryDataPath)

    def get_time_berry(time):
        try:
            a= str(time).split(' ')[2].split(':')
        except:
            a = ['nan','nan','nan']
        return a
    dfNewBerry['Hour'] = dfNewBerry['time'].apply(lambda x: int(get_time_berry(x)[0]))
    dfNewBerry['Minutes'] = dfNewBerry['time'].apply(lambda x: int(get_time_berry(x)[1]))
    dfNewBerry['Seconds'] = dfNewBerry['time'].apply(lambda x: int(get_time_berry(x)[2]))
    dfNewBerry = dfNewBerry[['Hour', 'Minutes', 'Seconds', 'spo2', 'pr']]
    dfNewBerry.columns = ['Hour', 'Minutes', 'Seconds', 'spo2_Berry', 'pr_Berry']
  
    newdf = pd.merge(dfNewBerry,dfNewApel,on=['Hour', 'Minutes', 'Seconds'])
    newdf = newdf.sort_values(by=['Hour', 'Minutes', 'Seconds'])
    newdf['TotalSec'] = newdf['Hour']*60*60 + newdf['Minutes']*60 + newdf['Seconds']
    initialSec = int(newdf['TotalSec'].min())
    newdf['TotalSec'] = newdf['TotalSec'] - initialSec
    newdf['x'] = newdf['x'] + 130
    newdf['y'] = newdf['y'] + 50
    newdf['z'] = newdf['z'] + 850
    newdf['xPer'] = newdf[['x']].pct_change()
    newdf['yPer'] = newdf[['y']].pct_change()
    newdf['zPer'] = newdf[['z']].pct_change()
    
    newdf = newdf.dropna()
    newdf['TotalChange'] = newdf['xPer'] + newdf['yPer'] + newdf['zPer'] 


    return newdf

newdf = insertNewData(PathTestApel, PathTestBerry)

SPO2errorDict = {}
HRerrorDict = {}

SPO2TrainColumns = ['SPO2_Apel','xPer','yPer','zPer', 'TotalChange']
HRTrainColumns = ['HR_Apel', 'xPer','yPer','zPer', 'TotalChange']

class LINEAR_REGRESSION:
    
    def SPO2_Prediction(df2predict,show = True,returnError = 0):
        regressionName = 'Linear'  
        X = df[SPO2TrainColumns]
        y = df[['spo2_Berry']].values
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)
        regressor = LinearRegression()
        regressor.fit(X_train,y_train)

        coeff_df = pd.DataFrame(regressor.coef_[0], X.columns, columns=['Coefficient'])
        #print(coeff_df)
        y_pred = regressor.predict(df2predict[SPO2TrainColumns])
        #print(y_pred)
        df2predict['SPO2_pred'] = y_pred
        
        if show == True:
            Support.showErrors(df2predict['SPO2_Apel'],df2predict['SPO2_pred'].values,df2predict['spo2_Berry'].values,regressionName,'SPO2')
            Support.plotSPO2(regressionName,df2predict,graphsFolderPath)
        return {'LinearReggresionSPO2': Support.showErrors(df2predict['SPO2_Apel'],df2predict['SPO2_pred'].values,df2predict['spo2_Berry'].values,regressionName,'SPO2',show = False)[returnError]}
    

    def HR_Prediction(df2predict,show = True,returnError = 0):
        regressionName = 'Linear'  
        X = df[HRTrainColumns]
        y = df[['pr_Berry']].values
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)
        regressor = LinearRegression()
        regressor.fit(X_train,y_train)

        coeff_df = pd.DataFrame(regressor.coef_[0], X.columns, columns=['Coefficient'])
        #print(coeff_df)
        y_pred = regressor.predict(df2predict[HRTrainColumns])
        #print(y_pred)
        df2predict['Pr_pred'] = y_pred
        
        if show == True:
            Support.showErrors(df2predict['HR_Apel'],df2predict['Pr_pred'].values,df2predict['pr_Berry'].values,regressionName,'Heart Rate')
            Support.plotHR(regressionName,df2predict,graphsFolderPath)
        return {'LinearRegressionHR':Support.showErrors(df2predict['HR_Apel'],df2predict['Pr_pred'].values,df2predict['pr_Berry'].values,regressionName,'Heart Rate',show=False)[returnError]}



class LOGISTIC_REGRESSION:
      
    def SPO2_Prediction(df2predict,show=True,returnError = 0):
        regressionName = 'Logistic'
        scale = StandardScaler()
        X = df[SPO2TrainColumns]
        scaledX = scale.fit_transform(X.values)
        y = df[['spo2_Berry']].values
        X_train, X_test, y_train, y_test = train_test_split(scaledX, y, test_size=0.2, random_state=0)
        regressor = LogisticRegression()
        regressor.fit(X_train,y_train)

        #coeff_df = pd.DataFrame(regressor.coef_[0], X.columns, columns=['Coefficient'])
        y_pred = regressor.predict(df2predict[SPO2TrainColumns])
        #print(y_pred)
        df2predict['SPO2_pred'] = y_pred
        if show == True:
            Support.showErrors(df2predict['SPO2_Apel'],df2predict['SPO2_pred'].values,df2predict['spo2_Berry'].values,regressionName,'SPO2')
            Support.plotSPO2(regressionName,df2predict,graphsFolderPath)
        return {'LogisticRegressionSPO2' : Support.showErrors(df2predict['SPO2_Apel'],df2predict['SPO2_pred'].values,df2predict['spo2_Berry'].values,regressionName,'SPO2',show=False)[returnError]}

    def HR_Prediction(df2predict,show=True,returnError = 0):
        regressionName = 'Logistic'
        X = df[HRTrainColumns]
        y = df[['pr_Berry']].values
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.00001, random_state=0)
        regressor = LogisticRegression()
        regressor.fit(X_train,y_train)

        coeff_df = pd.DataFrame(regressor.coef_[0], X.columns, columns=['Coefficient'])
        #print(coeff_df)
        y_pred = regressor.predict(df2predict[HRTrainColumns])
        #print(y_pred)
        df2predict['Pr_pred'] = y_pred
    
        if show == True:
            Support.showErrors(df2predict['HR_Apel'],df2predict['Pr_pred'].values,df2predict['pr_Berry'].values,regressionName,'Heart Rate')
            Support.plotHR(regressionName,df2predict,graphsFolderPath)
        return {'LogisticRegressionHR' :Support.showErrors(df2predict['HR_Apel'],df2predict['Pr_pred'].values,df2predict['pr_Berry'].values,regressionName,'Heart Rate',show=False)[returnError]}

class RIDGE_REGRESSION:

# Ridge Regression
    def SPO2_Prediction(df2predict,show = True,returnError = 0):
        regressionName = 'Ridge'
        from sklearn.linear_model import Ridge
        X = df[SPO2TrainColumns]
        y = df[['spo2_Berry']].values
        ridge = Ridge(alpha=Support.bestLambda(df,regressionName,returnError))
        ridge.fit(X,y)
        ridge_coef = ridge.coef_    
        ridge_intercept = ridge.intercept_
        y_pred = ridge.predict(df2predict[SPO2TrainColumns])
        #print(y_pred)
        df2predict['SPO2_pred'] = y_pred
        
        if show == True:
            Support.plotSPO2(regressionName,df2predict,graphsFolderPath)
            Support.showErrors(df2predict['SPO2_Apel'],df2predict['SPO2_pred'].values,df2predict['spo2_Berry'].values,'Ridge','SPO2')
            
        return {'RidgeRegressionSP02':Support.showErrors(df2predict['SPO2_Apel'],df2predict['SPO2_pred'].values,df2predict['spo2_Berry'].values,'Ridge','SPO2',show=False)[returnError]}

    def HR_Prediction(df2predict,show = True,returnError = 0):
        regressionName = 'Ridge'
        from sklearn.linear_model import Ridge
        X = df[['HR_Apel', 'x', 'y', 'z',  'Array']]
        y = df[['pr_Berry']].values
        ridge = Ridge(alpha=Support.bestLambda(df,regressionName,returnError))
        ridge.fit(X,y)
        ridge_coef = ridge.coef_    
        ridge_intercept = ridge.intercept_
        y_pred = ridge.predict(df2predict[HRTrainColumns])
        #print(y_pred)
        df2predict['Pr_pred'] = y_pred

        if show == True:
            Support.showErrors(df2predict['HR_Apel'],df2predict['Pr_pred'].values,df2predict['pr_Berry'].values,regressionName,'Heart Rate')
            Support.plotHR(regressionName,df2predict,graphsFolderPath)
        return {'RidgeRegressionHR':Support.showErrors(df2predict['HR_Apel'],df2predict['Pr_pred'].values,df2predict['pr_Berry'].values,regressionName,'Heart Rate',show=False)[returnError]}

class LASSO_REGRESSION:
    def SPO2_Prediction(df2predict,show = True,returnError = 0):
        regressionName = 'Lasso'
        from sklearn.linear_model import Lasso


        X = df[SPO2TrainColumns]
        y = df[['spo2_Berry']].values
        lasso = Lasso(alpha=Support.bestLambda(df,regressionName,returnError))
        lasso.fit(X,y)

        y_pred = lasso.predict(df2predict[SPO2TrainColumns])
        df2predict['SPO2_pred'] = y_pred
        if show ==True:
            Support.showErrors(df2predict['SPO2_Apel'],df2predict['SPO2_pred'].values,df2predict['spo2_Berry'].values,regressionName,'SPO2')
            Support.plotSPO2(regressionName,df2predict,graphsFolderPath)
        return {'LassoRegressionSPO2': Support.showErrors(df2predict['SPO2_Apel'],df2predict['SPO2_pred'].values,df2predict['spo2_Berry'].values,regressionName,'SPO2',show=False)[returnError]}

    def HR_Prediction(df2predict,show=True,returnError = 0):
        regressionName = 'Lasso'
        from sklearn.linear_model import Lasso
        X = df[HRTrainColumns]
        y = df[['pr_Berry']].values
        lasso = Lasso(alpha=Support.bestLambda(df,regressionName,returnError))
        lasso.fit(X,y)
         
        y_pred = lasso.predict(df2predict[HRTrainColumns])
        df2predict['Pr_pred'] = y_pred
        if show == True:
            Support.showErrors(df2predict['HR_Apel'],df2predict['Pr_pred'].values,df2predict['pr_Berry'].values,regressionName,'Heart Rate')
            Support.plotHR(regressionName,df2predict,graphsFolderPath)
        return {'LassoRegressionHR':Support.showErrors(df2predict['HR_Apel'],df2predict['Pr_pred'].values,df2predict['pr_Berry'].values,regressionName,'Heart Rate',show=False)[returnError]}

class SUPPORT_VECTOR_REGRESSION:
    def SPO2_Prediction(df2predict,show = True,returnError = 0):
        regressionName = 'SupportVectorRegression'
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = df[SPO2TrainColumns]
        y = df[['spo2_Berry']].values
        regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        

        y_pred = regr.fit(X, y).predict(df2predict[SPO2TrainColumns])
        df2predict['SPO2_pred'] = y_pred
        if show ==True:
            Support.showErrors(df2predict['SPO2_Apel'],df2predict['SPO2_pred'].values,df2predict['spo2_Berry'].values,regressionName,'SPO2')
            Support.plotSPO2(regressionName,df2predict,graphsFolderPath)
        return {'SupportVectorRegressionSPO2': Support.showErrors(df2predict['SPO2_Apel'],df2predict['SPO2_pred'].values,df2predict['spo2_Berry'].values,regressionName,'SPO2',show=False)[returnError]}

    def HR_Prediction(df2predict,show = True,returnError = 0):
        regressionName = 'SupportVectorRegression'
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = df[HRTrainColumns]
        y = df[['pr_Berry']].values
        regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        

        y_pred = regr.fit(X, y).predict(df2predict[SPO2TrainColumns])
        df2predict['Pr_pred'] = y_pred
        if show == True:
            Support.showErrors(df2predict['HR_Apel'],df2predict['Pr_pred'].values,df2predict['pr_Berry'].values,regressionName,'Heart Rate')
            Support.plotHR(regressionName,df2predict,graphsFolderPath)
        return {'SupportVectorRegressionHR':Support.showErrors(df2predict['HR_Apel'],df2predict['Pr_pred'].values,df2predict['pr_Berry'].values,regressionName,'Heart Rate',show=False)[returnError]}


'''
LOGISTIC_REGRESSION.SPO2_Prediction(newdf)
LOGISTIC_REGRESSION.HR_Prediction(newdf)

LINEAR_REGRESSION.SPO2_Prediction(newdf)
LINEAR_REGRESSION.HR_Prediction(newdf)

RIDGE_REGRESSION.SPO2_Prediction(newdf,returnError=1)
RIDGE_REGRESSION.HR_Prediction(newdf,returnError=1)

LASSO_REGRESSION.SPO2_Prediction(newdf,returnError=1)
LASSO_REGRESSION.HR_Prediction(newdf,returnError=1)
'''

SUPPORT_VECTOR_REGRESSION.SPO2_Prediction(newdf)
SUPPORT_VECTOR_REGRESSION.HR_Prediction(newdf)

'''

SPO2errorDict.update(LINEAR_REGRESSION.SPO2_Prediction(newdf,show=False,returnError=1))
SPO2errorDict.update(LOGISTIC_REGRESSION.SPO2_Prediction(newdf,show=False,returnError=1))
SPO2errorDict.update(RIDGE_REGRESSION.SPO2_Prediction(newdf,show=False,returnError=1))
SPO2errorDict.update(LASSO_REGRESSION.SPO2_Prediction(newdf,show=False,returnError=1))

print(SPO2errorDict)

HRerrorDict.update(LINEAR_REGRESSION.HR_Prediction(newdf,show=False,returnError=1))
HRerrorDict.update(LOGISTIC_REGRESSION.HR_Prediction(newdf,show=False,returnError=1))
HRerrorDict.update(RIDGE_REGRESSION.HR_Prediction(newdf,show=False,returnError=1))
HRerrorDict.update(LASSO_REGRESSION.HR_Prediction(newdf,show=False,returnError=1))

print(HRerrorDict)
'''