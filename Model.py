import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/Users/daniele/Desktop/Stock BTC Prediction/BTC-USD-all.csv', index_col=0)
df.drop(['Adj Close', 'Volume'], axis=1, inplace=True)
df.head()


# Train val test split
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
print(train_df.shape)
print(val_df.shape)
print(test_df.shape)
num_features = df.shape[1]
print(num_features)


# Min Max Scaler
train = train_df
scalers={}
for i in train_df.columns:
    scaler = MinMaxScaler(feature_range=(-1,1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    train[i]=s_s

test = test_df
for i in train_df.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+i] = scaler
    test[i]=s_s

val = val_df
for i in train_df.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(val[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+i] = scaler
    val[i]=s_s


# Function for window split
def split_series(series, n_past, n_future):
  #
  # n_past ==> no of past observations
  #
  # n_future ==> no of future observations
  #
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)


n_past = 6
n_future = 1
n_features = 4


X_train, y_train = split_series(train.values,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
X_val, y_val = split_series(val.values,n_past, n_future)
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1],n_features))
y_val = y_val.reshape((y_val.shape[0], y_val.shape[1], n_features))
X_test, y_test = split_series(test.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))


# Model
# E1D1
# n_features ==> no of features at each timestep in the data.
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)

encoder_states1 = encoder_outputs1[1:]

decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])

decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)

model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs1)

model_e1d1.summary()

# E2D2
# n_features ==> no of features at each timestep in the data.
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]

decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])

decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)

model_e2d2 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)

model_e2d2.summary()


reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=1,
    mode='auto', baseline=None, restore_best_weights=False
)
model_e1d1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
history_e1d1=model_e1d1.fit(X_train,y_train,epochs=25,validation_data=(X_test,y_test),batch_size=32,verbose=1,callbacks=[reduce_lr, early_stopping])
model_e2d2.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
history_e2d2=model_e2d2.fit(X_train,y_train,epochs=25,validation_data=(X_val,y_val),batch_size=32,verbose=1,callbacks=[reduce_lr, early_stopping])


plt.plot(history_e1d1.history['loss'])
plt.plot(history_e1d1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history_e2d2.history['loss'])
plt.plot(history_e2d2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


pred_e2d2=model_e2d2.predict(X_test)

print(pred_e2d2.shape)


X_prova = X_test
y_prova = []
pred = []
test = []

for _ in range(0,len(X_prova)):
    for i in X_prova[_]:
        y_prova.append(i[0])

    for i in pred_e2d2[_]:
        pred.append(i[0])

    for i in y_test[_]:
        test.append(i[0])

print(len(y_prova))
print(len(pred))
print(len(test))

plt.figure()
plt.scatter(list(range(0,len(y_prova))),y_prova)
plt.scatter(list(range(n_past,(len(pred)*n_past)+n_past, n_past)),pred)
plt.scatter(list(range(n_past,(len(test)*n_past)+n_past, n_past)),test)
plt.show()




reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=1,
    mode='auto', baseline=None, restore_best_weights=False
)

model_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(n_past, n_features)),
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(64, return_sequences=False),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=n_features),
    #tf.keras.layers.RepeatVector(n_future)
    tf.keras.layers.Reshape((1, 4))
])

model_lstm.summary()

model_lstm.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanAbsoluteError())
history_lstm=model_lstm.fit(X_train,y_train,epochs=100,validation_data=(X_val,y_val),batch_size=32,verbose=1,callbacks=[reduce_lr, early_stopping])

plt.plot(history_lstm.history['loss'])
plt.plot(history_lstm.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

pred_lstm=model_lstm.predict(X_test)

# Plot test prediction 
X_prova = X_test
y_prova = []
pred = []
test = []

for _ in range(0,len(X_prova)):
    for i in X_prova[_]:
        y_prova.append(i[3])

    for i in pred_lstm[_]:
        pred.append(i[3])

    for i in y_test[_]:
        test.append(i[3])

print(len(y_prova))
print(len(pred))
print(len(test))

plt.figure()
#plt.scatter(list(range(0,len(y_prova))),y_prova)
plt.plot(list(range(n_past,(len(pred)*n_past)+n_past, n_past)),pred)
plt.plot(list(range(n_past,(len(test)*n_past)+n_past, n_past)),test)
plt.legend(['y_pred', 'y_test'], loc='upper left')
plt.show()






# Multi LSTM Model
OUT_STEPS = 1
multi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_past, n_features)),
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*n_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape((OUT_STEPS, n_features))
])

multi_lstm_model.summary()

reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=1,
    mode='auto', baseline=None, restore_best_weights=False
)

multi_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
history_multi_lstm=multi_lstm_model.fit(X_train,y_train,epochs=100,validation_data=(X_val,y_val),batch_size=32,verbose=1,callbacks=[reduce_lr, early_stopping])

plt.plot(history_multi_lstm.history['loss'])
plt.plot(history_multi_lstm.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


pred_multi_lstm=multi_lstm_model.predict(X_test)

# Plot test prediction
X_prova = X_test
y_prova = []
pred = []
test = []

for _ in range(0,len(X_prova)):
    for i in X_prova[_]:
        y_prova.append(i[3])

    for i in pred_multi_lstm[_]:
        pred.append(i[3])

    for i in y_test[_]:
        test.append(i[3])

print(len(y_prova))
print(len(pred))
print(len(test))

plt.figure()
#plt.scatter(list(range(0,len(y_prova))),y_prova)
plt.plot(list(range(n_past,(len(pred)*n_past)+n_past, n_past)),pred)
plt.plot(list(range(n_past,(len(test)*n_past)+n_past, n_past)),test)
plt.legend(['y_pred', 'y_test'], loc='upper left')
plt.show()
