import re
from flask import Flask,request
from flask_jsonpify import jsonify
from random import randint
from flask_cors import CORS
import db
import pandas as pd



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
        f = pd.read_csv(file)
        columns = list(f.head(0))
        print(columns)
        print(file)
        return jsonify({
            "status":"Uploaded Successfully!..",
            "columns":columns
        }) 
    except Exception as e:
        print(e)
        return jsonify({"status":"Internal Server Error"}) 



if __name__ == '__main__':
    app.run(port=5002)