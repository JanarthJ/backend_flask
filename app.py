from flask import Flask,request

app = Flask(__name__)

import db#test to insert data to the data base

@app.route("/test")
def test():
    db.db.userCollection.insert_one({"name": "John"})
    return "Connected to the data base!"


@app.route('/')
def flask_mongodb_atlas():
    return "flask mongodb atlas!"

@app.route('/api/createuser', methods=['POST'])
def createUser():
    request_data = request.get_json()
    print(request_data)
    return "Successfully"

if __name__ == '__main__':
    app.run(port=5002)