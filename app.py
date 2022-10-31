from flask import Flask,request
from flask_jsonpify import jsonify
from random import randint
from flask_cors import CORS
import db
import pandas as pd
import matplotlib.pyplot as plt
# from statsmodels.tsa.arima_model import ARIMA
import warnings
import itertools
import datetime as dt
import math

from pandas.plotting import autocorrelation_plot
# from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from flask import Flask, render_template
from io import BytesIO
import base64



app = Flask(__name__)
CORS(app)


def token():
    tok=""
    for i in range(10):
        tok+=str(randint(0,9))
    return tok

#sample test api
@app.route('/')
def flask_mongodb_atlas():
    print("Called")
    return "flask mongodb atlas!"

@app.route('/plot')
def plot():
    img = BytesIO()
    y = [1,2,3,4,5]
    x = [0,2,1,3,4]

    plt.plot(x,y)

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    url="data:image/png;base64,"+str(plot_url)
    return jsonify({
        "status":"valid",
        "url":url
        })


#api for verification
@app.route('/checkAuth',methods=['POST'])
def checkAuth():
    print("verifyauth")
    data = request.get_json()
    print(data)    
    try:
        users=db.db.userCollection.find({"email": data["Email"],"token":data["token"]}) 
        output = [{'Email' : user['email']} for user in users] 
        if len(output)==1:
            return jsonify({"status":"Valid"})
        else:
            return jsonify({"status":"Notvalid"})  
    except Exception as e:
        print(e)
        return jsonify({"status":"Notvalid"})  

   

#user creation >>sign up api
@app.route('/createuser', methods=['POST'])
def createUser():
    request_data = request.get_json()
    print(request_data)
    users=db.db.userCollection.find({"email": request_data["email"]})
    output = [{'Name' : user['name'], 'EMail' : user['email']} for user in users]
    print(output)
    try:
        if len(output) > 0:
            return jsonify({"status":"EMail Already Exist"})
        else:            
            db.db.userCollection.insert_one(request_data) 
            print("created successfully")
            return jsonify({"status":"user created Successfully"})  
            
    except Exception as e:
        print(e)
        return jsonify({"status":"Server Error"})  

#api for login
@app.route('/auth',methods=['POST'])
def read():
    try:
        request_data = request.get_json()
        print(request_data)
        users = db.db.userCollection.find({"email":request_data["email"],"password":request_data["password"]})
        output = [{'name' : user['name'],'Email' : user['email'],'token':user['token']} for user in users]
        print(output)    
        if(len(output)==1):
            tok=token()
            print(tok)
            filt={"email":output[0]['Email']}
            updat = {"$set": {'token' : tok}}
            print(filt,updat)
            db.db.userCollection.update_one(filt,updat)
            print("Updated")
            output[0]['token']=tok            
            return jsonify({"status":"Verified","data":output})   
        else:
            return jsonify({"status":"Invalid credential"}) 
            
    except Exception as e:
        print(e)
        return jsonify({"status":"Server Error"})  


@app.route('/uploadcsv',methods=['POST'])
def uploadCsv():
    print("Called")
    try:
        file=request.files.get('file')
        ftrain = pd.read_csv(file)
        ftrain['SalesPerCustomer'] = ftrain['Sales']/ftrain['Customers']
        ftrain['SalesPerCustomer'].head()
        columns = list(ftrain.head(0))
        print(columns)
        targetPredictColumns=["Sales","Customers","SalesPerCustomer"]
        print(file)
        # print(hello(file))
        return jsonify({
            "status":"Uploaded Successfully!..",
            "columns":targetPredictColumns
        }) 
    except Exception as e:
        print(e)
        return jsonify({"status":"Internal Server Error"}) 


def predict():
    print("Pridict Called")
    try:
        import pandas as pd
        import numpy as np
        file=request.files.get('file')
        predictColumn=request.form.get('columnPredict')
        fromDate=request.form.get('fromDate')
        toDate=request.form.get('toDate')
        print(predictColumn,fromDate,toDate)
        # f = pd.read_csv(file)        
        # print(file)
        df = pd.read_csv(file)
        print(df)
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
        sales['Order Date'] = pd.to_datetime(sales['Order Date'], dayfirst=True)
        sales['Ship Date'] = pd.to_datetime(sales['Ship Date'], dayfirst=True)
        sales.info()

        print("CK1")
        sales = sales.loc[:, ["Order Date", "Sales", "Profit"]]

        #Set date column to index
        sales.set_index('Order Date', inplace = True)
        print(sales.info())
        sales = sales.groupby("Order Date").sum()
        sales
        print("CK2")
        # sales_by_year = pd.DataFrame()
        # sales_by_year
        print("CK21")
        # l=["2011","2012","2013","2014"]
        # try:
        #     # for year in l:
        #     #     print("loop")
        #     #     print("y=",year)
        #     #     temp_year = sales.loc[year, ["Sales"]]
        #     #     print("ty")
        #     #     temp_year.rename(columns={"Sales": year}, inplace = True)
        #     #     print("tymo")
        #     #     sales_by_year = pd.concat([sales_by_year, temp_year], axis=1)
        #     #     print("end")
        # except Exception as e:
        #     print(e)

        # sales_by_year
        print("CK3")

        monthly_sales = sales.groupby(pd.Grouper(freq='MS')).sum()

        print("CK4")

        from statsmodels.tsa.stattools import acf

        # Create Training and Test
        train = monthly_sales[["Sales"]][:36]
        test = monthly_sales[["Sales"]][36:]
        print("CK5")
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(monthly_sales["Sales"])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])

        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings("ignore")

        # order=(p, d, q)
        model = ARIMA(train["Sales"], order=(2, 1, 0))
        model_fit = model.fit()
        print(model_fit.summary())

        from sklearn.metrics import mean_squared_error
        y = test.values.astype('float32')

        X = train.values.astype('float32')

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
                    print("end sarimax")
                except:
                    continue
        
        print("End")

        mod = sm.tsa.statespace.SARIMAX(X,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

        results = mod.fit()
        print("result")

        import datetime as dt
        import math

        from pandas.plotting import autocorrelation_plot
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from sklearn.preprocessing import MinMaxScaler


        pd.options.display.float_format = '{:,.2f}'.format
        np.set_printoptions(precision=2)
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        print("end res")


        train_start_dt = '2011-11-01 00:00:00'
        test_start_dt = '2014-11-01 00:00:00'

        scaler = MinMaxScaler()
        train['load'] = scaler.fit_transform(train)
        train.head(10)
        print("res11")
        train
        df['prediction'] = train['load']
        out = df[['Order Date',"prediction"]]
        out
        print(out)
        print("ended")
        # out.to_csv("D:\kaar_jj\submission.csv", index = False)





        



















        
        # out.to_csv("/home/janarthanan/submission.csv", index = False)
        # print("this is hello fun")   

        return jsonify({
            "status":"Predicted Successfully!.."
        })     
    except Exception as e:
        print(e)
        return jsonify({"status":"Internal Server Error"}) 



@app.route('/predict',methods=['POST'])
def forecast_predict():
    try:
        from prophet import Prophet
        file=request.files.get('file')
        targetColumn=request.form.get('columnPredict')
        dayPredict=request.form.get('dayPredict')
        print(targetColumn,dayPredict)
        print(type(targetColumn),type(dayPredict))
        train = pd.read_csv(file)
        # print(len(train))
        columns = list(train.head(0))
        print(columns)
        # train.head(5)

        train['SalesPerCustomer'] = train['Sales']/train['Customers']
        train['SalesPerCustomer'].head()   
        train = train.dropna()
        print("CK1")
        sales = train[train.Store == 1].loc[:, ['Date', targetColumn]]
        print("CK2")
        # reverse to the order: from 2013 to 2015
        sales = sales.sort_index(ascending = False)

        sales['Date'] = pd.DatetimeIndex(sales['Date'])
        sales.dtypes
        print("CK3")
        sales = sales.rename(columns = {'Date': 'ds',targetColumn: 'y'})
        print("CK4")
        sales_prophet = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True)
        sales_prophet.fit(sales)
        print("CK5")
        # Make a future dataframe for 6weeks
        sales_forecast = sales_prophet.make_future_dataframe(periods=int(dayPredict), freq='D')
        # Make predictions
        sales_forecast = sales_prophet.predict(sales_forecast)
        
        print("CK6")
        print(len(sales_forecast))        
        print("CK7")
        # matplotlib parameters
        matplotlib.rcParams['axes.labelsize'] = 18
        matplotlib.rcParams['xtick.labelsize'] = 14
        matplotlib.rcParams['ytick.labelsize'] = 14
        matplotlib.rcParams['text.color'] = 'k'
        img = BytesIO()
        sales_prophet.plot(sales_forecast, xlabel = 'Date', ylabel = 'targetColumn')
        # plt.title('Drug Store Forecasting',fontsize=18, color= 'green', fontweight='bold')
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        url="data:image/png;base64,"+str(plot_url)
        print("CK8")
        return jsonify({
            "status":"Uploaded Successfully!..",
            "graph":url
        }) 
    except Exception as e:
        print(e)
        return jsonify({"status":"Internal Server Error"}) 

if __name__ == '__main__':
    app.run(port=5002)