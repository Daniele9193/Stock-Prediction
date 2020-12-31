import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

tf.version.VERSION

df = pd.read_csv('/Users/daniele/Desktop/Stock BTC Prediction/BTC-USD-all.csv', index_col=0)
df.head()

df.describe()


df.isna().sum()

df.dtypes

fig = px.line(df, x=df.index, y=["Open","Close",'High','Low'], title='BTC-USD Stock')
fig.update_layout(
    autosize=False,
    #width=2000,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
        )
    )

fig.show()

fig = px.line(df, x=df.index, y='Volume', title='BTC-USD Volume')
fig.update_layout(
    autosize=False,
    width=1000,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
        )
    )

fig.show()



df.drop(['Volume'],1,inplace=True)
df.drop(['Adj Close'],1,inplace=True)


column_indices = {name: i for i, name in enumerate(df.columns)}

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

# Normalization
scaler = MinMaxScaler()
scaler.fit(train_df)
train_df = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)
val_df = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns)
test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
train_df.head()

# Class WindowGenerator
class WindowGenerator():
    def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, test_df=test_df, label_columns = None):
        # Store the raw data
        self.train_df =  train_df
        self.val_df =  val_df
        self.test_df =  test_df

        # Work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_column_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]


    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


w1 = WindowGenerator(input_width=6, label_width=1, shift=1, label_columns=['Close'])
w1

# Function split_window
def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.label_slice, :]

    if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
        print(labels)

    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

WindowGenerator.split_window = split_window



# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                           np.array(train_df[100:100+w1.total_window_size]),
                           np.array(train_df[200:200+w1.total_window_size])])


example_inputs, example_labels = w1.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')

train_df.iloc[:,0:3]

# Function make_dataset
def make_dataset(self, data):
    data = np.array(data, dtype=np.float64)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, shuffle=True, batch_size=32)
    ds = ds.map(self.split_window)
    return ds

WindowGenerator.make_dataset = make_dataset



@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test


w1.train.element_spec
