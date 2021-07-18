import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from reliability.Distributions import Weibull_Distribution
from reliability.Probability_plotting import plot_points
from reliability.Fitters import Fit_Weibull_3P
from reliability.Fitters import Fit_Weibull_2P

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

def count_by_month(df):
    df=df.copy()
    df['ym']=df['start'].dt.to_period('M')
    cdf=df.groupby('ym').size()
    cdf=cdf.reset_index()
    cdf.columns=['ym', 'number']
    cdf.ym=cdf.ym.dt.to_timestamp()
    return(cdf)

cdf=count_by_month(df)

st.title('TTE Analysis on California WildFires')
st.write("Time-to-event (TTE) data is unique because the outcome of interest is not only whether or not an event occurred, but also when that event occurred.")
st.write("In this application, I will focus on time to next California WildFires. The application will show:  a.The distribution of time interval between two fire incidents will be display. b.The trend of fire accidents. c.The seasonal of fire accidents.")
st.write("The table below shows start time, end time, location (contry) and time to next incidents of every California WildFires incidents between 2013 and 2020.")

st.write(df)

st.write("The Figure below show California WildFires incidents by month.")

fig=plt.figure()
ax=sns.lineplot(data=cdf,
             x='ym',
             y='number')
plt.xlabel('')
plt.ylabel('Fires incidents by month')
st.pyplot(fig)

st.write("Decompose time series to trend, seasonality noise.")
result=seasonal_decompose(cdf['number'], 
                          model='additive', 
                          period=12)
fig=result.plot()
st.pyplot(fig)
st.write("Both seasonal and trends exist. There are more and moree fire incidents with time advances. And more incidents happen in summer.")


st.write("The Figure below show distribution of time to next California WildFires incidents.")
fig=plt.figure()
ax=sns.kdeplot(df.t)
plt.xlim(0,150)
st.pyplot(fig)

st.write("The Figure below show time to next California WildFires incidents vs start time.")
fig=plt.figure()
ax=sns.scatterplot(data=df,
                   x='start',
                   y='t')
st.pyplot(fig)

st.write("Fit time to next California WildFires incidents to Weibull distribution .")
t=df.t.values
fig=plt.figure()
weibull_fit = Fit_Weibull_2P(failures=t,show_probability_plot=False,print_results=False)
weibull_fit.distribution.SF(label='Fitted Distribution',color='steelblue')
plot_points(failures=t,func='SF',label='failure data',color='red',alpha=0.7)
plt.legend()
st.pyplot(fig)
st.write("It looks fit well.")
st.write("The Weibull distribution is similar to the exponential distribution. While the exponential distribution assumes a constant hazard, the Weibull distribution assumes a monotonic hazard that can either be increasing or decreasing but not both. It has two parameters. The shape parameter (σ ) controls whether hazard increases (σ<1 ) or decreases (σ>1 ) (in the exponential distribution, this parameter is set to 1). The scale parameter, (1/σ)exp(-β0/σ), determines the scale of this increase/decrease. Since the Weibull distribution simplifies to the exponential distribution when σ=1, the null hypothesis that σ=1 can be tested using a Wald test. The main advantage of this model is that it is both a PH and AFT model, so both hazard ratios and time ratios can be estimated. Again, the main drawback is that the assumption of monotonicity of the baseline hazard may be implausible in some cases.")



