import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

url='https://raw.githubusercontent.com/xieli2021/web_test/main/California_Fire_Incidents%20(1).csv'

def read_clean(url):
    dfR=pd.read_csv(url)
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
    return(df)

df=read_clean(url)
st.write(df)

