import re
from flask import Flask,request
from flask_jsonpify import jsonify
from random import randint
app = Flask(__name__)
import db

def token():
    tok=""
    for i in range(10):
        tok+=str(randint(0,9))
    return tok

#sample test api
@app.route('/')
def flask_mongodb_atlas():
    return "flask mongodb atlas!"

#api for verification
@app.route('/checkAuth',methods=['POST'])
def verify():
    data = request.get_json()
    print(data)    
    try:
        users=db.db.userCollection.find({"email": data["email"],"token":data["token"]}) 
        output = [{'Email' : user['email']} for user in users] 
        if len(output)==1:
            return jsonify({"status":"Valid"})
        else:
            return jsonify({"status":"Notvalid"})  
    except:
        return jsonify({"status":"Notvalid"})  

   

#user creation >>sign up api
@app.route('/api/createuser', methods=['POST'])
def createUser():
    request_data = request.get_json()
    print(request_data)
    users=db.db.userCollection.find({"email": request_data["email"]})
    output = [{'Name' : user['name'], 'EMail' : user['email']} for user in users]
    print(output)
    try:
        if len(output) > 0:
            return "EMail already found"
        else:            
            db.db.userCollection.insert_one(request_data) 
            print("created successfully")
            return jsonify({"status":"user created Successfully"})  
            
    except:
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
            
    except:
        return jsonify({"status":"Server Error"})  

if __name__ == '__main__':
    app.run(port=5002)