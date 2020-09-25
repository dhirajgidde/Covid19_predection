import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import plotly.graph_objects as go
import plotly.express as px

lr_rate=1.0e-4

df=pd.read_csv("case_time_series.csv")

print(df)

# fig = px.line(df, x = 'Date', y = 'Daily Confirmed', title='Covid-19 India')
# fig.show()

# plt.figure(figsize=(16,8))
# plt.title("Covid19 Growth in India")
# plt.plot(df['Daily Confirmed'])
# plt.xlabel('Daily Confirmed')
# plt.ylabel('Daily Confirmed')
# plt.show()

data=df.filter(['Total Confirmed'])

dataset=data.values

training_data_len=math.ceil(len(dataset)*.8)

scaler=MinMaxScaler(feature_range=(0,1))

scaled_data=scaler.fit_transform(dataset)

train_data=scaled_data[0:training_data_len, : ]

x_train=[]
y_train=[]
print("WE are here")
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i , 0])

    y_train.append(train_data[i,0])

adam=Adam(lr=lr_rate)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print("Lets Begin")
model=Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50,return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer=adam,loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs=10)


test_data=scaled_data[training_data_len-60: , : ]

x_test=[]
y_test=dataset[training_data_len : , : ]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test=np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
print("Pred Price: ",predictions)

rmse=np.sqrt(np.mean(((predictions+y_test)**2)))

print("Almost finshed")
#print(rmse)

#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


print(valid)

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Covid 19 Future Cases')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Total Confirmed', fontsize=18)
plt.plot(train['Total Confirmed'])
plt.plot(valid[['Total Confirmed', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# fig = px.line(df, x = 'Date', y = valid[['Daily Confirmed']], title='Covid-19 India')
# fig.show()
