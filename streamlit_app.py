import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

st.title('My first and Test app')

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

dfR=pd.read_csv('https://raw.githubusercontent.com/xieli2021/web_test/main/California_Fire_Incidents%20(1).csv')
df=dfR.copy()
df=df.loc[:,['Counties','Started','Extinguished']]
df.columns=['country','start','end']
df.start=pd.to_datetime(df.start)
df.end=pd.to_datetime(df.end)
df=df.sort_values('start')
df=df.loc[df.start>'2013-01-01',:]
df['t']=0
df=df.reset_index(drop=True)
for i in np.r_[1:df.shape[0]]:
    df.loc[i,'t']=(df.start[i]-df.start[i-1])/np.timedelta64(1, 'D')
print(df)
