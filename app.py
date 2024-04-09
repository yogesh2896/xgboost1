def activate(c1,c2):

# XGBoost Algorithm
  import pandas as pd
  import streamlit as st
  import numpy as np
  from xgboost import XGBRegressor
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import r2_score
  from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
  import matplotlib.pyplot as plt
  import seaborn as sns
  import matplotlib.pyplot as plt
  import plotly.express as px

  df=pd.read_csv(url)
  df=df[[c1,c2]]
  df['date'] = pd.to_datetime(df[c1]).dt.to_period('M').dt.start_time
  df=df.groupby('date').sum("Sales_Value")
  df.reset_index(inplace=True)
  df.columns=['date','Sales_Value']
  q1=np.percentile(df['Sales_Value'],25)
  q3=np.percentile(df['Sales_Value'],75)
  iqr=q3-q1
  ll=q1-1.5*iqr

  ul=q3+1.5*iqr
  df=df[(df['Sales_Value']>ll) & (df['Sales_Value']<ul)]








  df['month']=df['date'].dt.month
  df['year']=df['date'].dt.year
  df['weekday']=df['date'].dt.weekday
  df['day']=df['date'].dt.day
  df['dayofweek']=df['date'].dt.day_of_week
  x=df.drop(['Sales_Value','date'],axis=1)
  y=df['Sales_Value']
  xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
  l=round(len(df)/3)
# xtrain=x[:-l]
# xtest=x[-l:]
# ytrain=y[:-l]
# ytest=y[-l:]
  params = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 2, 3, 4],
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
  }
  model=XGBRegressor()
  randomized_search = RandomizedSearchCV(model, params, cv=5, n_jobs=-1, verbose=2)
  randomized_search.fit(xtrain, ytrain)
  best_params = randomized_search.best_params_
  model=XGBRegressor(n_estimators=best_params['n_estimators'],min_child_weight=best_params['min_child_weight'],max_depth=best_params['max_depth'],
                   learning_rate=best_params['learning_rate'],gamma=best_params['gamma'],colsample_bytree=best_params['colsample_bytree'])
  model=model.fit(xtrain,ytrain)
  ypred=model.predict(xtest)
  comp_df=pd.DataFrame(ypred,ytest)
  comp_df.reset_index(inplace=True)
  comp_df.columns=['predicted','actual']
  print('Below prediction based on XGBoost Regressor Algorithm')
  # plt.plot(comp_df['predicted'], label='Predicted')
  # plt.plot(comp_df['actual'], label='Actual')
  # plt.xlabel('Value')
  # plt.ylabel('Frequency')
  # _ = plt.legend()
  # plt.show()
  # comp_df.reset_index(inplace=True)
  # comp_df.set_index('index',inplace=True)
  # # print(comp_df)
  # import plotly.express as px
  # fig=px.line(comp_df,x=comp_df['index'],y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  # # st.write(fig)
  # fig.show()
  # print(comp_df)
  # sns.lineplot(x=comp_df.index,y=comp_df['predicted'])
  # sns.lineplot(x=comp_df.index,y=comp_df['actual'])
  # plt.show()
  fig=px.line(comp_df,x=comp_df.index,y=['predicted','actual'],color_discrete_sequence=px.colors.qualitative.Plotly)
  st.write(fig)
  score=r2_score(comp_df['actual'],comp_df['predicted'])*100
  st.write("The above model accuracy score:",score,"%")
  # fig.show()

  # sdate=pd.to_datetime(sdate)
  # edate=pd.to_datetime(edate)
  # sdate='02/02/2024'
  # edate='02/04/2024'
  pred_df=pd.date_range(start=sdate,end=edate)
  pred_df=pd.DataFrame(pred_df)
  pred_df.columns=['date']
  pred_df['month']=pred_df['date'].dt.month
  pred_df['year']=pred_df['date'].dt.year
  pred_df['weekday']=pred_df['date'].dt.weekday
  pred_df['day']=pred_df['date'].dt.day
  pred_df['dayofweek']=pred_df['date'].dt.day_of_week
  # st.dataframe(pred_df)
  x=pred_df.drop(['date'],axis=1)
  ypred1=model.predict(x)
  newdf=pd.DataFrame(ypred1,pred_df['date'])
  newdf.columns=['Prediction']
  pfig=px.line(newdf,x=newdf.index,y='Prediction')
  st.write(pfig)
  st.dataframe(newdf)








  
import streamlit as st
st.title("Welcome to Dynamic Predictive Model by Enoah Isolution")
c1=st.text_input('Enter the date column name')
c2=st.text_input('Enter the target column name')
url=st.file_uploader(label='Upload your data')
sdate=st.date_input(label='select start date')
edate=st.date_input(label='select end date')





if st.button('Forecasting'):


  activate(c1,c2)


