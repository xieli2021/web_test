import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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



st.title('TTE Analysis on California WildFires')
st.write("Time-to-event (TTE) data is unique because the outcome of interest is not only whether or not an event occurred, but also when that event occurred.")
st.write("In this application, I will focus on time to next California WildFires. The application will show:  a.The distribution of time interval between two fire incidents will be display. b.The trend of fire accidents. c.The seasonal of fire accidents.")
st.write("The table below shows start time, end time, location (contry) and time to next incidents of every California WildFires incidents between 2013 and 2020.")

st.write(df)

st.write("The Figure below show distribution of time to next California WildFires incidents.")
fig=plt.figure()
ax=sns.kdeplot(df.t)
plt.xlim(0,150)
st.pyplot(fig)

st.write("The Figure below show time to next California WildFires incidents vs start time.")
st.write("It seems seasonal exist but no trends.")
fig=plt.figure()
ax=sns.scatterplot(data=df,
                   x='start',
                   y='t')
st.pyplot(fig)
