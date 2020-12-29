import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

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
