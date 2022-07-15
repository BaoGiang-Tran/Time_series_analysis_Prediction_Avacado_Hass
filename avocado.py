from asyncio import futures
from click import option
import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet 
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import r2_score
from math import sqrt
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric 
from prophet.plot import plot_plotly
import plotly.offline as py
import plotly.express as px
from PIL import Image
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import base64

#1. Read Data
#Import data
data=pd.read_csv('avocado_new.csv')
image1 = Image.open('image1.png')


   

#GUI
st.image(image1)
st.title("Data Science Project ")
st.header("Hass avocado price prediction")
def sidebar_bg(side_bg):
    
   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
side_bg = 'image2_new.png'
sidebar_bg(side_bg)

#Upload file
upload_file = st.file_uploader('Choose a file', type = ['csv'])
if upload_file is not None:
    data = pd.read_csv(upload_file)
    data.to_csv('avocado_newdata.csv', index = False)
#-----------------------------------------------------------------------
#Visualization
q_averageprice = data.groupby(by = ['quarter'])['AveragePrice'].mean()



#2. Load Data
input = data[['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargeBags','type']]
output = data[['AveragePrice']]

#3. Scale Data
scaler = RobustScaler()
df_scaler = scaler.fit_transform(input[['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargeBags']])
df_scaler = pd.DataFrame(df_scaler,columns=['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargeBags'])

encoder = LabelEncoder()
input['type_encoder'] = encoder.fit_transform(input['type'])
df_type = input['type_encoder']

#4. Buil Model - ExtraTreeRegressor
X = pd.concat([df_scaler,df_type],axis=1)
y = data['AveragePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state=12)
ex = ExtraTreesRegressor()
model = ex.fit(X_train,y_train)

#5.Evaluate Model
yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)
R_square_train = round(model.score(X_train,y_train),3)
R_square_test = round(model.score(X_test,y_test),3)
RMSE = round(mean_squared_error(y_test,yhat_test,squared=False),3)

#-----------------------------------------------------------------------------------------
#6. Pre_data for facebookprophet

#5.Facebook Prophet Model
#Model
cali_cv = pd.read_csv('cali_cv_fb.csv')
cali_cv['ds'] = pd.to_datetime(cali_cv['ds'])
fbp_model = Prophet(interval_width=0.95,
                           weekly_seasonality=False,
                           daily_seasonality=False,
                           yearly_seasonality=True,
                           changepoint_prior_scale=0.001,
                           seasonality_prior_scale=0.1,
                           holidays_prior_scale=0.01,
                           changepoint_range=0.8,
                           growth='linear',
                           seasonality_mode='multiplicative'
                           )

cali_con_model = fbp_model.fit(cali_cv)

cali_or= pd.read_csv('cali_or_fb.csv')
cali_or['ds'] = pd.to_datetime(cali_or['ds'])
fbp_model1 = Prophet(interval_width=0.95,
                           weekly_seasonality=False,
                           daily_seasonality=False,
                           yearly_seasonality=True,
                           changepoint_prior_scale=0.001,
                           seasonality_prior_scale=0.1,
                           holidays_prior_scale=0.01,
                           changepoint_range=0.8,
                           growth='linear',
                           seasonality_mode='multiplicative'
                           )

cali_or_model = fbp_model1.fit(cali_or)

west_cv = pd.read_csv('west_cv_fb.csv')
west_cv['ds'] = pd.to_datetime(west_cv['ds'])
fbp_model2 = Prophet(interval_width=0.95,
                           weekly_seasonality=False,
                           daily_seasonality=False,
                           yearly_seasonality=True,
                           changepoint_prior_scale=0.001,
                           seasonality_prior_scale=0.1,
                           holidays_prior_scale=0.01,
                           changepoint_range=0.8,
                           growth='linear',
                           seasonality_mode='multiplicative'
                           )

west_model = fbp_model2.fit(west_cv)

sc_cv = pd.read_csv('sc_cv_fb.csv')
sc_cv['ds'] = pd.to_datetime(sc_cv['ds'])
fbp_model3 = Prophet(interval_width=0.95,
                           weekly_seasonality=False,
                           daily_seasonality=False,
                           yearly_seasonality=True,
                           changepoint_prior_scale=0.001,
                           seasonality_prior_scale=0.1,
                           holidays_prior_scale=0.01,
                           changepoint_range=0.8,
                           growth='linear',
                           seasonality_mode='multiplicative'
                           )

sc_model = fbp_model3.fit(sc_cv)

north_or = pd.read_csv('north_or_fb.csv')
north_or['ds'] = pd.to_datetime(north_or['ds'])
fbp_model4 = Prophet(interval_width=0.95,
                           weekly_seasonality=False,
                           daily_seasonality=False,
                           yearly_seasonality=True,
                           changepoint_prior_scale=0.001,
                           seasonality_prior_scale=0.1,
                           holidays_prior_scale=0.01,
                           changepoint_range=0.8,
                           growth='linear',
                           seasonality_mode='multiplicative'
                           )
north_model = fbp_model4.fit(north_or)




#---------------------------------------------------------------------------------
#Gui
st.sidebar.header('Table of Content')
menu = ['Business Objective','Statistic & Analysis','Build Model','End-user Application']
choice = st.sidebar.selectbox('Content', menu)
if choice == 'Business Objective':
    st.header('Business Objective')
    st.subheader('Business Understand')
    st.write('Avehass is a company incorporated in the city of Uruapan, Michoacan, Mexico; dedicated to the production, marketing, product selection and packaging of the highest quality avocado, since we are located in the area of higher production and world class recognition.')
    st.write('The company has a need to build avocado price prediction model that is a technology tool to support the development of strategies and plans to develop avocado farms in different regions.')
    
    st.write('Purpose of project:')
    st.write(' - Building avocado average price prediction model - Regression.')
    st.write(' - Determine stretagic development region and avocado price forecast - Time Series Model.')
    
    st.write('Scope of project: ')
    st.write("- Data is selected directly from retailers' cash registers and retail scanners on a weekly basis (total volumne and price)")
    
    st.subheader('Data Understand')
    st.write('- Average Price: price/unit (avocado)')
    st.write('- Product Lookup codes : PLUs (4046, 4225,4770)')
    
    
    st.subheader('Acquire')
    st.write('Problem 1: ')
    st.write("- USA s Avocado AveragePrice Prediction (Regression như Linear Regression, Random Forest, XGB Regressor...)")
    st.write('Problem 2: ')
    st.write('- Conventional/Organic Avocado Average Price Prediction for the future in California/NewYork... - Time Series ARIMA, Prophet..')
    
    st.subheader('Conclusion')
    st.write('From statistical results and analysis according to average price, volumne and saless show that:')
    st.write('- From 2015 to 2017, the price of conventional and organic avocado grew steadily year by year.')
    st.write('- With conventional avocado, California, West and SouthCentral have higher volumne and revenue compared to other regions.')
    st.write("- On the orderhand, volumne and sales of organic avocado mainly come from 3 regions California , West and NorthEast. However, NorthEast's revenue has grown significantly faster than the other regions (in 2017 it increased by 91 percenatge compared to 2015).")
    st.write("The project team applied ExtraTreeRegressor Model that is selected based on the Lazypredict result to predict the average avocado price. R square train = 1.0 and R square test = 0.78. This model is enough good.")
    st.write("Time Series Model - Apply Facebook Prophet after comparing the results with ARIMA and HotWinter Model. fbpprophet model has been applied to predict price avocado at California (conventional and organic), West - Conventional, SouthCentral - Conventional, and Northeast - Organic")
    st.write("with perfomance of fbpprophet model, the project team used cross validation function of Prophet Model to evalute performance. Mean absolute percent error (MAPE) of this model equal 10%.")
    
if choice =='Statistic & Analysis':
    st.header('Statistic & Analysis')
#Show data
    st.subheader("1.Data")
    if st.checkbox('Preview Dataset'):
        st.text('First 5 lines of dataset')
        st.dataframe(data.head())
        st.text('Last 5 lines of dataset')
        st.dataframe(data.tail())
    if st.checkbox('Info of dataset'):
        st.text('Describle of dataset')
        st.code(data.describe())
        
#Visualization Data
#Avg Price
    st.subheader('2.Data Visualization')
    st.write('2.1. Average Price')
    st.caption('Average Price by Quarter in Year from 2015 to 2018')
        
    q_avg = q_averageprice.plot(grid = True, figsize = (14,7), title = 'Average Price by Quarter in Year')
    st.pyplot(q_avg.figure)
     
    type_averageprice = data.groupby(by = ['type','quarter'])['AveragePrice'].mean().reset_index()
    type_averageprice['quarter'] = type_averageprice['quarter'].astype(str)
    f = plt.figure(figsize = (14,7))
    sns.lineplot(x = type_averageprice['quarter'], y = type_averageprice['AveragePrice'], hue = type_averageprice['type']).set_title('Average Price by quater & type')
    plt.show()
    st.pyplot(f.figure)
        
    st.write('2.2. Total Volumne')
    
    type_volumne = data.groupby(by = ['type','quarter'])['TotalVolumne'].sum().reset_index()
    type_volumne['quarter'] = type_volumne['quarter'].astype(str)
    f1 = plt.figure(figsize = (15,5))
    sns.barplot(x = type_volumne['quarter'], y = type_volumne['TotalVolumne'], hue = type_volumne['type'])
    plt.title('Total Volumne by type & quater in year')
    plt.show()
    st.pyplot(f1.figure)
    
    region_ = data.groupby(by = ['region','year'])['TotalVolumne'].sum().reset_index()
    region_.sort_values(by ='TotalVolumne', ascending = False, inplace = True)
    TotalUS = region_[region_['region']=='TotalUS']
    f2 = plt.figure(figsize = (15,5))
    sns.barplot(x = TotalUS['year'], y = TotalUS['TotalVolumne'], color = 'b').set_title('Total Volumne by Year in US (2015 - 2018)')
    plt.show()
    st.pyplot(f2.figure)
    
    state_big = region_.set_index('region')
    state_big = state_big.loc[['West','California','SouthCentral', 'Northeast','Southeast','GreatLakes', 'Midsouth', 'LosAngeles', 'Plains']]
    state_big.reset_index(inplace = True)
    f3 = plt.figure(figsize = (15,5))
    sns.barplot(x = state_big['region'], y = state_big['TotalVolumne'], hue = state_big['year']).set_title('Total Volumne by Year in State (2015 - 2018)')
    plt.show()
    st.pyplot(f3.figure)
    
    st.write('2.3. Sales')
    data['sales'] = data['TotalVolumne']*data['AveragePrice']
    region_sales = data.groupby(by = ['region','year','type'])['sales'].sum().reset_index()
    region_sales.sort_values(by = 'sales', ascending = False, inplace = True)
    region_sales.year = pd.to_datetime(region_sales.year, format='%Y')
    region_sales['year'] = pd.to_datetime(region_sales['year']).dt.year
    state_big_sales = region_sales.set_index('region')
    state_big_sales = state_big_sales.loc[['West','California','SouthCentral', 'Northeast','Southeast','GreatLakes', 'Midsouth', 'LosAngeles', 'Plains']]
    state_big_sales.reset_index(inplace = True)
    sales_con = state_big_sales[state_big_sales['type'] == 'conventional']
    sales_org = state_big_sales[state_big_sales['type'] == 'organic']
    
    f4 = plt.figure(figsize=(15,6))
    sns.lineplot(x = sales_con['year'], y = sales_con['sales'], hue = sales_con['region']).set_title('Sales of Conventional by year & region')
    plt.show()
    st.pyplot(f4.figure)
    
    f5 = plt.figure(figsize=(15,6))
    sns.lineplot(x = sales_org['year'], y = sales_org['sales'], hue = sales_org['region']).set_title('Sales of Organic by year & region')
    plt.show()
    st.pyplot(f5.figure)
          
#Build Model
if choice =='Build Model':
    st.subheader('3. Build Model')

    #Peformance of Model
    st.write('3.1. ExtraTreeRegressor - Average Price Prediction')
    st.write('Evalute model performance')
    st.write('- R_square_train:',R_square_train )
    st.write('- R_square_test:',R_square_test)
    st.write('- RMSE:',RMSE)
    
    f7 = plt.figure(figsize=(10,5))
    plt.scatter(yhat_test, y_test)
    plt.xlabel('Model predictions')
    plt.ylabel('True values')
    plt.plot([0, 3], [0, 3], 'r-')
    plt.title('Evaluate Actual Value & Predict Value')
    plt.show()
    st.pyplot(f7.figure)
    
    f8 = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.kdeplot(y_train, color='blue', label='True train values')
    sns.kdeplot(yhat_train, color='red', label='Predict train values')
    plt.title('y_train vs yhat_train')
    plt.subplot(1,2,2)
    sns.kdeplot(y_test, color='blue', label='True test values')
    sns.kdeplot(yhat_test, color='red', label='Predict test values')
    plt.title('y_test vs yhat_test')
    plt.legend()
    plt.show()
    st.pyplot(f8.figure)

    st.write('3.2. Facebook Prophet - Strategic Region')
    select_region = st.selectbox('Strategic Development Region',['California','West','SouthCentral','NorthEast'])
    if select_region == 'California':
        st.write('Conventional at California')
        cv_cali_con = cross_validation(cali_con_model, initial = '170 days',period = '30 days', horizon='365 days',parallel='processes')
        cali_con_pm = performance_metrics(cv_cali_con)
        st.code(cali_con_pm.head())
        st.code(cali_con_pm.tail())
        cv1 = plot_cross_validation_metric(cv_cali_con, metric = 'mape')
        st.pyplot(cv1)
        
        st.write('Organic at California')
        cv_cali_or = cross_validation(cali_or_model, initial = '170 days',period = '30 days', horizon='365 days',parallel='processes')
        cali_or_pm = performance_metrics(cv_cali_or)
        st.code(cali_or_pm.head())
        st.code(cali_or_pm.tail())
        cv2 = plot_cross_validation_metric(cv_cali_or, metric = 'mape')
        st.pyplot(cv2)
        
    if select_region == 'West':
        st.write('Conventional at West')
        cv_west = cross_validation(west_model, initial = '170 days',period = '30 days', horizon='365 days',parallel='processes')
        west_pm = performance_metrics(cv_west)
        st.code(west_pm.head())
        st.code(west_pm.tail())
        cv3 = plot_cross_validation_metric(cv_west, metric = 'mape')
        st.pyplot(cv3)
        
    if select_region =='SouthCentral':
        st.write('Conventional at SouthCentral')
        cv_sc = cross_validation(sc_model, initial = '170 days',period = '30 days', horizon='365 days',parallel='processes')
        sc_pm = performance_metrics(cv_sc)
        st.code(sc_pm.head())
        st.code(sc_pm.tail()) 
        cv4= plot_cross_validation_metric(cv_sc, metric = 'mape')
        st.pyplot(cv4)
        
    if select_region =='NorthEast':
        st.write('Organic at NorthEast')
        cv_north = cross_validation(north_model, initial = '170 days',period = '30 days', horizon='365 days',parallel='processes')
        north_pm = performance_metrics(cv_north)
        st.code(north_pm.head())
        st.code(north_pm.tail())
        cv5 = plot_cross_validation_metric(cv_north, metric = 'mape')
        st.pyplot(cv5)
        
#End User Addition
if choice == 'End-user Application':      
    st.header('End-user Application')
    model_pre = ['Avg Price Prediction','Time Series FacebookProphet']
    choice_model = st.sidebar.selectbox('Model Prediction', model_pre)
    if choice_model == 'Avg Price Prediction':
        st.write('End-user Input Feature')
        #Collects user input feature into data
        uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type=['csv'])
        #Choose Upload or Input data
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
        else:
            def user_input_data():
                Total_Volumne = st.number_input('Total Volumne', min_value=0, value=0)
                PLU_4046 = st.number_input('PLU 4046', min_value=0, value=0)
                PLU_4225 = st.number_input('PLU 4225', min_value=0, value=0)
                PLU_4770 = st.number_input('PLU 4770', min_value=0, value=0)
                Total_Bags = st.number_input('Total Bags', min_value=0, value=0)
                Small_Bags = st.number_input('Small Bags', min_value=0, value=0)
                Large_Bags = st.number_input('Large Bags',min_value=0, value=0)
                Types = st.selectbox('Types',('conventional', 'organic'))
                data_new = {'TotalVolumne': Total_Volumne,
                            '4046': PLU_4046,
                            '4225': PLU_4225,
                            '4770':PLU_4770,
                            'TotalBags':Total_Bags,
                            'SmallBags':Small_Bags,
                            'LargerBags':Large_Bags,
                            'types':Types}
                data_input = pd.DataFrame(data_new, index=[0])
                columns = ['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargerBags','types']
                data_input = data_input.reindex(columns = columns)
                return data_input
            with st.sidebar.form('my_form'):
                input_df = user_input_data()
                sub = st.form_submit_button('Submit')
        #Scale and encoder data       
        scale = scaler.transform(input_df[['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargerBags']])
        scale = pd.DataFrame(scale,columns=['TotalVolumne','4046','4225','4770','TotalBags','SmallBags','LargerBags'])
        input_df['type_encoder'] = encoder.transform(input_df['types'])
        lb_encoder = input_df['type_encoder']
        X_new = pd.concat([scale,lb_encoder],axis=1)
        #Show prediction
        st.dataframe(input_df)  
        prediction = model.predict(X_new)
        st.write('Prediction:', prediction)
    elif choice_model == 'Time Series FacebookProphet':
        #Select input or upload data
        st.write('End-user Input Feature')
        ts = st.sidebar.number_input('Time prediction', min_value=0, value=0, max_value=10)
        
        #Select model 
        region_model = ['California Convetional Prediction', 'California Organic Prediction','West Conventional Prediction','SouthCentral Conventional Prediction','Northeast Organic Prediction']
        choice_region = st.sidebar.selectbox('Region', region_model)
        
        if choice_region =='California Convetional Prediction':
            ds = cali_con_model.make_future_dataframe(periods=48*ts, freq='W')
            forecast = cali_con_model.predict(ds)
            new_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]
            
            fig = cali_con_model.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(),cali_con_model,forecast)
            st.pyplot(fig.figure)
            st.dataframe(new_forecast)
            
        elif choice_region =='California Organic Prediction':
            ds = cali_or_model.make_future_dataframe(periods=48*ts, freq='W')
            forecast = cali_or_model.predict(ds)
            new_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]
            
            fig = cali_or_model.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(),cali_or_model,forecast)
            st.pyplot(fig.figure)
            st.dataframe(new_forecast)
            
        elif choice_region == 'West Conventional Prediction':
            ds = west_model.make_future_dataframe(periods=48*ts, freq='W')
            forecast = west_model.predict(ds)
            new_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]
            
            fig = west_model.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(),west_model,forecast)
            st.pyplot(fig.figure)
            st.dataframe(new_forecast)
            
        elif choice_region =='SouthCentral Conventional Prediction':
            ds = sc_model.make_future_dataframe(periods=48*ts, freq='W')
            forecast = sc_model.predict(ds)
            new_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]
            
            fig = sc_model.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(),sc_model,forecast)
            st.pyplot(fig.figure)
            st.dataframe(new_forecast)
            
        elif choice_region == 'Northeast Organic Prediction':
            ds = north_model.make_future_dataframe(periods=48*ts, freq='W')
            forecast = north_model.predict(ds)
            new_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]
            fig = north_model.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(),north_model,forecast)
            st.pyplot(fig.figure)
            st.dataframe(new_forecast)