import pandas as pd
import matplotlib.pyplot as plt

dfApel = pd.read_json('/home/anastasios/Documents/ElectricalEngeneeringMasterDegree/MedicalProject/NewColumnsApel.json').transpose()
dfApel =dfApel[[ 'timestamp','HR', 'SPO2',  'xdata', 'ydata', 'zdata']]

dfApel = dfApel.dropna(subset=['timestamp'])
def get_time(time):
    try:
        a= str(time).split(' ')[1].split(':')
    except:
        a = ['nan','nan','nan']
    return a
dfApel['Hour'] = dfApel['timestamp'].apply(lambda x: int(get_time(x)[0]))
dfApel['Minutes'] = dfApel['timestamp'].apply(lambda x: int(get_time(x)[1]))
dfApel['Seconds'] = dfApel['timestamp'].apply(lambda x: int(get_time(x)[2]))

def getMove(move):
    a = str(move).split(';')
    a.pop(-1)
    a = list(map(float, a))
    return sum(a)/len(a)
dfApel['x'] = dfApel['xdata'].apply(lambda x:getMove(x))
dfApel['y'] = dfApel['ydata'].apply(lambda x:getMove(x))
dfApel['z'] = dfApel['zdata'].apply(lambda x:getMove(x))
dfApel["Array"] = ((dfApel['x']**2)+(dfApel['y']**2)+(dfApel['z']**2))**1/2
dfApel = dfApel[[ 'HR', 'SPO2', 'Hour', 'Minutes', 'Seconds', 'x', 'y', 'z', 'Array']]
dfApel.columns = [ 'HR_Apel', 'SPO2_Apel', 'Hour', 'Minutes', 'Seconds', 'x', 'y', 'z', 'Array']
dfBerry = pd.read_csv('/home/anastasios/Documents/ElectricalEngeneeringMasterDegree/MedicalProject/NewColumnBerry.csv')

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
print(df.columns)
print(df['SPO2_Apel'])
plt.plot(df['TotalSec'],df['spo2_Berry'])
plt.plot(df['TotalSec'],df['SPO2_Apel'])
#plt.show()
