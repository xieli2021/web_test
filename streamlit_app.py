# -*- coding: utf-8 -*-
"""
Name: LI XIE
CS605: Section 1
Data: California Fire incidents data
URL: https://share.streamlit.io/xieli2021/web_test/main
Description:
This program focus on time to event (TTE) data analysis. The application will show: 
a.	The distribution of time interval between two fire incidents will be display.
b.	The trend of fire accidents.
c.	The seasonal of fire accidents.
"""

import streamlit as st
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

#from reliability.Fitters import Fit_Everything
#fig=plt.figure()
#results = Fit_Everything(failures=t)
#st.pyplot(fig)

class time_series_simulator():
    
    def __init__(self):
        '''
        initialize
        '''
        pass
    
    def white_noise(self,length,mean_wn=0,variance_wn=4,radius_wn=2,
                   distribution_wn='Guassian',plot=False):
        '''
        Simulate white noise
        '''
        if distribution_wn=='Guassian':
            wn=np.random.normal(loc=mean_wn,
                               scale=variance_wn,
                               size=length)
        elif distribution_wn=='Uniform':
            wn=np.random.uniform(low=-radius_wn,
                                up=radius_wn,
                                size=length)
        if plot:
            plt.plot(wn)
            plt.figure()
        return(wn)
    
    def poisson(self,length,lam=1011,plot=False):
        '''
        Simulate Poisson distribution
        '''
        pois=np.random.poisson(lam=lam, size=length)
        if plot:
            plt.plot(pois)
            plt.figure()
        return(pois)

    def random_walk(self,length,first_value_rw=0,drift_rw=0,
                    mean_wn=0,variance_wn=4,radius_wn=2,
                    distribution_wn='Guassian',
                    plot=False,constraint_positive=False):
        '''
        Simulate white noise
        If drift is not equal to 0, include trend
        '''
        wn=self.white_noise(length=length,mean_wn=mean_wn,variance_wn=variance_wn,
                            radius_wn=radius_wn,distribution_wn=distribution_wn)
        rw=np.zeros(length)
        rw[0]=first_value_rw
        for i in np.r_[1:length]:
            rw[i]=first_value_rw+np.sum(wn[range(i)])+drift_rw*i
            if constraint_positive:
                if rw[i]<0:
                    rw[i]=0
        if plot:
            plt.plot(rw)
            plt.figure()
        return(rw)

    def linear_trend(self,length,slope=1,first_value=0):
        '''
        Simulate simple linear trend
        '''
        ts=np.r_[0:length]
        ts=first_value+ts*slope
        return(ts)

    def auto_regression(self,length,first_value_ar=0,slope=1,constant=1,
                        mean_wn=0,variance_wn=4,radius_wn=2,
                        distribution_wn='Guassian',plot=False,
                        constraint_positive=False):
        '''
        Simulate autoregression proccess
        '''
        wn=self.white_noise(length=length,mean_wn=mean_wn,variance_wn=variance_wn,
                            radius_wn=radius_wn,distribution_wn=distribution_wn)
        ar=np.zeros(length)
        ar[0]=first_value_ar
        for i in np.r_[1:length]:
            ar[i]=ar[i-1]*slope+constant+wn[i]
            if constraint_positive:
                if ar[i]<0:
                    ar[i]=0
        if plot:
            plt.plot(ar)
            plt.figure()
        return(ar)

    def diff_days(self,first_date,last_date):
        '''
        Calculate difference of days between first and last date
        '''
        first_date=pd.to_datetime(first_date)
        last_date=pd.to_datetime(last_date)
        days=(last_date-first_date).days
        return(days)
    
    def convert_df(self,first_date,ts,title='',file_name='',
                   constraint_int=True,plot=True,write=False):
        '''
        Convert to dataframe
        '''
        df=np.zeros([len(ts),3])
        df=pd.DataFrame(df)
        df.columns=['ds','days','y']
        df.ds=pd.to_datetime(first_date)
        df.days=range(len(ts))
        df.days=pd.to_timedelta(df.days,unit='D')
        df.ds=df.ds+df.days
        df.y=ts
        df=df.loc[:,['ds','y']]
        if constraint_int:
            df.y=df.y.astype(int)
        if plot:
            df.plot.line(x='ds',y='y',
                         legend=False,
                         title=title)
            plt.xlabel('')
            plt.figure()
        if write:
            write_time=str('a')[:19].replace(' ','-').replace(':','-')
            file_name=file_name+write_time+'.csv'
            df.to_csv(file_name,index=False)
        return(df)

    def comprise(self,first_date,component_s,method='additive',file_name='',
                 title='',plot=True,return_df=True,write=True):
        '''
        Comprise several components
        '''
        length=len(component_s[0])
        for i in np.r_[1:len(component_s)]:
            if len(component_s[i])!=length:
                print('Warning!!! Error in length!!!')
        if method=='additive':
            ts=np.zeros(length)
            for component in component_s:
                ts=ts+component
        elif method=='multiplicative':
            ts=np.repeat(1,length)
            for component in component_s:
                ts=ts*component
        df=self.convert_df(first_date=first_date,ts=ts,file_name=file_name,
                           title=title,plot=plot,write=write)
        if return_df:
            return(df)
        else:
            return(ts)
        
    def discount(self,df,discount_start_date,discount_end_date,
                 discount_increase_ratio=1.6,
                 discount_increase_amount=2000,
                 method='multiplicative',file_name='',
                 title='',plot=True,return_df=True,write=True):
        discount_start_date=pd.to_datetime(discount_start_date)
        discount_end_date=pd.to_datetime(discount_end_date)
        df['discount_or_not']='normal'
        df.loc[(df.ds>discount_start_date)&
               (df.ds<discount_end_date),'discount_or_not']='discount'
        if method=='additive':
            df.loc[(df.ds>discount_start_date)&
                   (df.ds<discount_end_date),'y']=df.loc[(df.ds>discount_start_date)&
                                                         (df.ds<discount_end_date),'y']+discount_increase_amount
        elif method=='multiplicative':
            df.loc[(df.ds>discount_start_date)&
                   (df.ds<discount_end_date),'y']=df.loc[(df.ds>discount_start_date)&
                                                         (df.ds<discount_end_date),'y']*discount_increase_ratio
        if plot:
            sns.scatterplot(data=df,
                             x='ds',
                             y='y',
                             hue='discount_or_not')
            plt.xlabel('')
            plt.figure()
        if write:
            write_time=str('a')[:19].replace(' ','-').replace(':','-')
            file_name=file_name+write_time+'.csv'
            df.to_csv(file_name,index=False)
        if return_df:
            return(df)
        else:
            return(df.y.values)
    
    def holiday(self,df,holiday_md=['-01-01','-12-25','-05-01'],
                 holiday_increase_ratio=1.6,
                 holiday_increase_amount=2000,
                 method='multiplicative',file_name='',
                 title='',plot=True,return_df=True,write=True):
        df['holiday_or_not']='normal'
        # get unique year list
        year_unique=set(df.ds.dt.year)
        y=[]
        for i in year_unique:
            y.append(str(i))
        y_c=np.repeat(y,len(holiday_md))
        holiday_c=np.tile(holiday_md,len(y))
        holiday_all=[]
        # get all holiday in all year
        for i in range(len(holiday_c)):
            h=str(y_c[i])+holiday_c[i]
            h=pd.to_datetime(h)
            holiday_all.append(h)
        df.loc[np.isin(df.ds,holiday_all),'holiday_or_not']='holiday'
        
        if method=='additive':
            df.loc[np.isin(df.ds,holiday_all),'y']=df.loc[np.isin(df.ds,holiday_all),'y']+\
                holiday_increase_amount
        elif method=='multiplicative':
            df.loc[np.isin(df.ds,holiday_all),'y']=df.loc[np.isin(df.ds,holiday_all),'y']*holiday_increase_ratio
        if plot:
            sns.scatterplot(data=df,
                             x='ds',
                             y='y',
                             hue='holiday_or_not')
            plt.xlabel('')
            plt.figure()
        if write:
            write_time=str('a')[:19].replace(' ','-').replace(':','-')
            file_name=file_name+write_time+'.csv'
            df.to_csv(file_name,index=False)
        if return_df:
            return(df)
        else:
            return(df.y.values)
        
first_date='2012-01-01'
last_date='2021-01-01'

tss=time_series_simulator()
ts_len=tss.diff_days(first_date,last_date)
ts_len

fig=plt.figure()
ts_basic=tss.random_walk(length=ts_len,first_value_rw=10,drift_rw=0,
                         mean_wn=0,variance_wn=11,plot=True)
st.pyplot(fig)
