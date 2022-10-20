from flask import Flask
from flask_pymongo import pymongo
from app import app


CONNECTION_STRING = "mongodb+srv://senary:senary2020@userauthcluster.p0wbp.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('flask_mongodb_atlas')
user_collection = pymongo.collection.Collection(db, 'user_collection')