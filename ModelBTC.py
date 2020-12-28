import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv('/Users/daniele/Desktop/Stock BTC Prediction/BTC-USD.csv')
df.head()


fig = px.line(df, x="Date", y="Close", title='BTC-USD Stock')
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
