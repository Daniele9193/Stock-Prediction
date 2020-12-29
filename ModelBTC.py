import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/Users/daniele/Desktop/Stock BTC Prediction/BTC-USD-all.csv', index_col=0)
#df['Date'] = pd.to_datetime(df['Date'])
df.head()

df.describe()


df.isna().sum()

df.dtypes

fig = px.line(df, x=df.index, y=["Open","Close",'High','Low'], title='BTC-USD Stock')
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

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index)
df_scaled.columns = df.columns
df_scaled.head()

df_scaled.drop(['Adj Close', 'Volume'], axis=1, inplace=True)
df_scaled.columns

df_scaled.describe()
