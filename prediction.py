from flask_jsonpify import jsonify
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
import warnings
import itertools
import datetime as dt
import math

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler


def hello(file):
    print(file)
    print("predict1")
    df = pd.read_csv(file)
    columns = list(df.head(0))
    print(columns)
    df.head(10)
    df.shape
    df.info()
    df.describe()
    df.isnull().sum()
    sales = df.copy()
    sales = sales.drop("Postal Code", axis = 1) #axis=1 for column
    sales.info()


    # In[16]:


    sales.isnull().sum().sum()


    # In[17]:


    #convert datetime columns to datetime objects
    sales['Order Date'] = pd.to_datetime(sales['Order Date'], dayfirst=True)
    sales['Ship Date'] = pd.to_datetime(sales['Ship Date'], dayfirst=True)
    sales.info()

    print("CK1")
    # In[105]:





    # In[40]:





    # In[107]:


    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)


    # In[108]:


    test['load'] = scaler.fit_transform(test)


    # In[109]:


    test


    # In[110]:


    train.shape


    # In[111]:


    test.shape


    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[18]:


    #Get only order date, sales, and profit
    sales = sales.loc[:, ["Order Date", "Sales", "Profit"]]

    #Set date column to index
    sales.set_index('Order Date', inplace = True)


    # In[19]:


    #confirm we have a Datetime series object
    print(sales.info())


    # In[20]:


    ax = sales.plot()
    plt.gcf().set_size_inches(15, 10)
    ax.set_ylabel("Dollars (USD)")
    plt.show()


    # In[21]:


    sales = sales.groupby("Order Date").sum()
    sales


    # In[23]:


    #Create empty dataframe to store yearly data
    sales_by_year = pd.DataFrame()

    for year in ["2011", "2012", "2013", "2014"]:
        temp_year = sales.loc[year, ["Sales"]].reset_index(drop = True)
        temp_year.rename(columns={"Sales": year}, inplace = True)
        sales_by_year = pd.concat([sales_by_year, temp_year], axis=1)

    sales_by_year


    # In[24]:


    daily_sales = sales.groupby(pd.Grouper(freq='D')).sum()
    daily_sales.plot()
    plt.gcf().set_size_inches(15, 10)
    plt.ylabel("Dollars (USD)")
    plt.show()


    # In[25]:


    monthly_sales = sales.groupby(pd.Grouper(freq='MS')).sum()
    monthly_sales.plot()
    plt.gcf().set_size_inches(15, 10)
    plt.title("Sales with MS frequency")
    plt.ylabel("Dollars (USD)")
    plt.show()


    # In[21]:


    #export files
    #daily_sales = daily_sales[["Sales"]]
    #monthly_sales = monthly_sales[["Sales"]]
    #yearly_sales = yearly_sales[["Sales"]]
    #daily_sales.to_csv("/content/drive/MyDrive/Grad School/INFO 659/SuperstoreSalesPredictor/data/daily_sales.csv", header=True)
    #monthly_sales.to_csv("/content/drive/MyDrive/Grad School/INFO 659/SuperstoreSalesPredictor/data/monthly_sales.csv", header=True)
    #yearly_sales.to_csv("/content/drive/MyDrive/Grad School/INFO 659/SuperstoreSalesPredictor/data/yearly_sales.csv", header=True)


    # In[60]:


    from statsmodels.tsa.stattools import acf

    # Create Training and Test
    train = monthly_sales[["Sales"]][:36]
    test = monthly_sales[["Sales"]][36:]


    # In[27]:


    from statsmodels.tsa.stattools import adfuller

    result = adfuller(monthly_sales["Sales"])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])


    # In[28]:


    import numpy as np
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(3, 2)

    #create plot
    axes[0, 0].plot(train["Sales"]); axes[0, 0].set_title('Original Series')
    plot_acf(train["Sales"], ax=axes[0, 1], title=" Autocorrelation");
    axes[1, 0].plot(train["Sales"].diff().dropna()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(train["Sales"].diff().dropna(), ax=axes[1, 1], title="Autocorrelation");
    axes[2, 0].plot(train["Sales"].diff().diff().dropna()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(train["Sales"].diff().diff().dropna(), ax=axes[2, 1], title="Autocorrelation");

    #visuals
    fig.suptitle('Autocorrelation using 30D Rolling Average', weight='bold')
    axes[0,0].set_ylabel("Sales (USD)")
    axes[1,0].set_ylabel("")
    axes[2,0].set_ylabel("")
    axes[0,1].set_xticks([])
    axes[1,1].set_xticks([])
    axes[2,1].set_xticks([])
    plt.gcf().set_size_inches(30, 15)
    plt.subplots_adjust(hspace=.5)
    plt.show()


    # In[29]:


    
    warnings.filterwarnings("ignore")

    # order=(p, d, q)
    model = ARIMA(train["Sales"], order=(2, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())


    # In[30]:


    model_fit.plot_predict(dynamic=False)
    plt.ylabel("Sales (USD)")
    plt.gcf().set_size_inches(15, 10)
    plt.show()


    # In[31]:


    from sklearn.metrics import mean_squared_error
    y = test.values.astype('float32')

    X = train.values.astype('float32')


    # In[32]:


    # create a differenced series
    def difference(X, interval=1):
        diff = list()
        for i in range(interval, len(X)):
            value = X[i] - X[i - interval]
            diff.append(value)
        return diff
    
    # invert differenced value
    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]


    # In[33]:


    history = [x for x in X] #Create a list of all training data
    months_in_year = 12
    predictions=list()
    yhat = float(model_fit.forecast()[0])
    yhat = inverse_difference(history, yhat, months_in_year)
    predictions.append(y[0])
    history.append(y[0])
    print('Predicted: {yhat:.3f}, Expected: {y[0]:.3f}') #output first month's performance

    #loop through all months and calculate performance
    for i in range(1, len(y)):
        months_in_year = 12
        diff = difference(history, months_in_year) #account for seasonality
        model = ARIMA(diff, order=(2, 1, 0)) #create ARIMA model
        model_fit = model.fit() #fit model
        yhat = model_fit.forecast()[0] #get prediction value
        yhat = inverse_difference(history, yhat, months_in_year) #reverse the difference to get value
        predictions.append(yhat) #add predicted value to list
        # observation
        obs = y[i]
        history.append(obs)
        print('Predicted: %.3f, Expected: %.3f' % (yhat, obs))
    
    # report performance
    rmse = np.sqrt(mean_squared_error(y, predictions))
    print('\nRMSE: %.3f' % rmse)


    # In[34]:


    #plot actual versus forecast
    plt.plot(y, label="Actual")
    plt.plot(predictions, color='red', label="Prediction")

    #visuals
    plt.title("Sales Predictions for 2014")
    plt.xlabel("Month in 2014")
    plt.xticks(ticks=np.arange(0,12,1), labels=["Jan", "Feb", "Mar", "Apr", "May", "June", 
                                                "July", "Aug", "Sept", "Oct", "Nov", "Dec"])
    plt.ylabel("Sales (USD)")
    plt.gcf().set_size_inches(15, 10)
    plt.legend(loc="upper left")
    plt.show()


    # In[35]:


    # Define the d and q parameters to take any value between 0 and 1
    import statsmodels.api as sm
    q = d = range(0, 2)
    # Define the p parameters to take any value between 0 and 3
    p = range(0, 4)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


    # In[ ]:





    # In[36]:


    warnings.filterwarnings("ignore") # specify to ignore warning messages

    AIC = []
    SARIMAX_model = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(X,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()

                print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
                AIC.append(results.aic)
                SARIMAX_model.append([param, param_seasonal])
            except:
                continue


    # In[48]:


    # Let's fit this model
    mod = sm.tsa.statespace.SARIMAX(X,
                                    order=SARIMAX_model[AIC.index(min(AIC))][0],
                                    seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    import datetime as dt
    import math

    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler


    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore") # specify to ignore warning messages


    # In[61]:


    train_start_dt = '2011-11-01 00:00:00'
    test_start_dt = '2014-11-01 00:00:00'


    # In[62]:


    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    df['prediction'] = train['load']
    out = df[['Order Date',"prediction"]]
    out.to_csv("/home/janarthanan/submission.csv", index = False)

    return "returned"
    