import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/home/anastasios/Documents/ElectricalEngeneeringMasterDegree/MedicalProject/medicalDevice2.csv' )

df['index'] = [i for i in range(len(df.iloc[:,0]))]
def divideBy10(index):
    if (index%10 == 0) and (index != 0):
        return True
    else:
        return False
df['Drop'] = df['index'].apply(lambda x: divideBy10(x))
GBlist = []
count = 1
for i in range(len(df['Drop'].values)):
    if df['Drop'].values[i] == True:
        count += 1
    else:
        x=5
    GBlist.append(count)

df['Order'] = GBlist

df = df[['Order','spo2', 'pr']]
df = df.groupby(['Order']).agg({'spo2': 'mean', 'pr':'mean'})
df['Order'] = df.index
df = df[['Order','spo2', 'pr']]
df.columns = ['Order','spo2_valid', 'pr_valid']
df = df.rename_axis(None)
#print(df)
'''
df = df.loc[df['Drop'] == True]
df['Order'] = df['index']/10
df = df[['Order','spo2', 'pr']]
df.columns = ['Order','spo2_valid', 'pr_valid']
'''

dfApel = pd.read_json('/home/anastasios/Documents/ElectricalEngeneeringMasterDegree/MedicalProject/Apel2.json').transpose()

def getMove(move):
    a = str(move).split(';')
    a.pop(-1)
    a = list(map(float, a))
    return sum(a)/len(a)
dfApel['x'] = dfApel['xdata'].apply(lambda x:getMove(x))
dfApel['y'] = dfApel['ydata'].apply(lambda x:getMove(x))
dfApel['z'] = dfApel['zdata'].apply(lambda x:getMove(x))
dfApel["Array"] = ((dfApel['x']**2)+(dfApel['y']**2)+(dfApel['z']**2))**1/2

def getOrder(Measurements):
    return str(Measurements).split('Measurements')[1]
dfApel['index1'] = dfApel.index
dfApel['Order'] = dfApel['index1'].apply(lambda x: int(getOrder(x)))
dfApel = dfApel.sort_values(by=['Order'])
dfApel = dfApel[['Order','HR', 'SPO2','x','y','z','Array']]

df = df.loc[df['Order'] <= dfApel['Order'].max()]
df = pd.merge(df,dfApel,on=['Order'])

'''plt.plot(df['Order'],df['x'], label= 'x')
plt.plot(df['Order'],df['y'], label= 'y')
plt.plot(df['Order'],df['z'], label= 'z')
plt.plot(df['Order'],df['Array'], label= 'Array')

plt.legend()
#plt.show()'''

'''
plt.plot(df['Order'],df['pr_valid'], label= 'Medical Device')
plt.plot(df['Order'],df['HR'], label = 'Apel Device')
plt.plot(df['Order'],df['spo2_valid'], label= 'Medical Device2')
plt.plot(df['Order'],df['SPO2'], label = 'Apel Device2')
plt.legend()

#plt.show()
'''

print(df.columns)
'''
####################################################################################
# PREDICT SPO2
####################################################################################

X = df[['SPO2', 'x', 'y', 'z','Array']]
y = df[['spo2_valid']].values
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train,y_train)

coeff_df = pd.DataFrame(regressor.coef_[0], X.columns, columns=['Coefficient'])
#print(coeff_df)
y_pred = regressor.predict(df[['SPO2', 'x', 'y', 'z','Array']])
#print(y_pred)
df['SPO2_pred'] = y_pred
print(df)


plt.plot(df['Order'],df['spo2_valid'], label= 'Medical Device2')
plt.plot(df['Order'],df['SPO2'], label = 'Apel Actual')
plt.plot(df['Order'],df['SPO2_pred'], label = 'Apel Predicted')
plt.title('SPO2')
plt.legend()
plt.show()
'''
####################################################################################
# PREDICT HEART RATE
####################################################################################
X = df[['HR', 'x', 'y', 'z','Array']]
y = df[['pr_valid']].values
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train,y_train)

coeff_df = pd.DataFrame(regressor.coef_[0], X.columns, columns=['Coefficient'])
#print(coeff_df)
y_pred = regressor.predict(df[['HR', 'x', 'y', 'z','Array']])
#print(y_pred)
df['HR_pred'] = y_pred
print(df)
plt.plot(df['Order'],df['pr_valid'], label= 'Medical Device2')
plt.plot(df['Order'],df['HR'], label = 'Apel Actual')
plt.plot(df['Order'],df['HR_pred'], label = 'Apel Predicted')
plt.title('Heart Rate')
plt.legend()
plt.show()


'''X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]


y = df[['quality']].values
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

coeff_df = pd.DataFrame(regressor.coef_[0], X.columns, columns=['Coefficient'])
print(X_test[1])
X_test[1] = [8,0.82,0, 4.1,0.095,6,14, 0.99854, 3.4,0.53,9.6]
y_pred = regressor.predict(X_test[0:2])
print(y_pred[1])'''



