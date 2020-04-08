# Recurrent Neural Network

# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#GPU Training
import keras
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


# Importing the training set
pjme = pd.read_csv('AEP_hourly.csv')
pjme['Datetime'] = pjme['Datetime'].map(pd.to_datetime)
def get_dom(dt):
    return dt.day
pjme['date'] = pjme['Datetime'].map(get_dom)

def get_weekday(dt):
    return dt.weekday()
pjme['weekday'] = pjme['Datetime'].map(get_weekday)

def get_hour(dt):
    return dt.hour
pjme['hour'] = pjme['Datetime'].map(get_hour)

def get_year(dt):
    return dt.year
pjme['year'] = pjme['Datetime'].map(get_year)

def get_month(dt):
    return dt.month
pjme['month'] = pjme['Datetime'].map(get_month)

def get_dayofyear(dt):
    return dt.dayofyear
pjme['dayofyear'] = pjme['Datetime'].map(get_dayofyear)

def get_weekofyear(dt):
    return dt.weekofyear
pjme['weekofyear'] = pjme['Datetime'].map(get_weekofyear)



# Feature Scaling
training_set = pjme.iloc[:10000, 1:2]
testing_set =  pjme.iloc[10000:12000, 1:2]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
training_set['AEP_MW'] = sc.fit_transform(training_set['AEP_MW'].values.reshape(-1,1))
training_set_scaled = training_set.values
testing_set['AEP_MW'] = sc.fit_transform(testing_set['AEP_MW'].values.reshape(-1,1))
testing_set_scaled = testing_set.values


# Creating a data structure with 400 timesteps and 1 output
X_train = []
y_train = []
for i in range(400, 10000):
    X_train.append(training_set_scaled[i-400:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)   
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 1, batch_size = 128)


#Saving Model
#Pickle
import pickle
filename = "model.pkl"
pickle.dump(regressor, open(filename, 'wb'))



# Part 3 - Making the predictions and visualising the results
X_test = []
y_test = []
for i in range(400, 2000):
    X_test.append(testing_set_scaled[i-400:i, 0])
    y_test.append(testing_set_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)   
# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, y_test)
#print(result)


import sklearn.preprocessing
from sklearn.metrics import r2_score
rnn_predictions = regressor.predict(X_test)
rnn_score = r2_score(y_test,rnn_predictions)
print("R2 Score of RNN model = ",rnn_score)

def plot_predictions(test, predicted, title):
    plt.figure(figsize=(16,4))
    plt.plot(test, color='blue',label='Actual power consumption data')
    plt.plot(predicted, alpha=0.7, color='orange',label='Predicted power consumption data')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized power consumption scale')
    plt.legend()
    plt.show()
    
plot_predictions(y_test, rnn_predictions, "Predictions made by simple RNN model")