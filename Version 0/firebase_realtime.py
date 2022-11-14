# pip install pyrebase4

import pyrebase
import random

config = {
    'apiKey': "xxx",
    'authDomain': "xxx",
    'databaseURL': "xxx",
    'projectId': "xxx",
    'storageBucket': "xxx",
    'messagingSenderId': "xxx",
    'appId': "xxx",
    'measurementId': "xxx"
}

firebase = pyrebase.initialize_app(config)

# Realtime Database
db = firebase.database()       # realtime database

# Storage for Images
# WARNING!! - free plan has limited capacity per day
# storage = firebase.storage()   

number = random.randint(0,100)

test_data = {
    "Number" : '{:.2f}'.format(number)
}

# send data to realtime database
db.child("test_data").push(test_data)   # create parent and child

db.update(test_data)

# send data to firebase storage
# storage.child('_____.jpg').put(image)