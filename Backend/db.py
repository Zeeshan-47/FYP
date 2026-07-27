import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.getenv("uri")

client = MongoClient(mongo_uri, server_api=ServerApi('1'))

print(client.server_info())
db = client["crime-o-graphy"]
users_collection = db["users"]
reports_collection = db["crime-reports"]
crime_categories_collection = db["crime_categories"]
alerts_collection = db["alerts"]