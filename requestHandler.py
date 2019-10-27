from flask import Flask, request, abort, jsonify
from blobAdder import *
from numpy import random
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import blobAdder
from machineLearning import trainDataset

from blobAdder import *
from detectFace import *

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def handle_push():
    # if not request.json:
    #     abort(400)
    import json

    if request.method == 'POST':
        base = request.form['base64']
        print(base)
        file = convertToFile(base)
        #container name has to be lowercase
        container = 'expressiveblobs'
        createContainer(container)
        addBlob(file, container)
        val = detect_faces_url("https://bosehomehub.blob.core.windows.net/" + container + "/" + file)
        return jsonify({'status': val}), 201

    if request.method == 'GET':
        print("helllo")
        urls = urlList('expressiveblobs')
        emotMap = computeAverage(urls)
        print(emotMap)
        emo = determineEmotion(emotMap)
        print(emo)
        deleteCont('expressiveblobs')
        return jsonify({'like': emo}), 202

    # if request.method == 'GET':
    #     print("helllo")
    #     cart = trainDataset()
    #     urls = urlList('expressiveblobs')
    #     emotMap = computeAverage(urls)
    #     print(emotMap)
    #     #emo = determineEmotion(emotMap)
    #     #print(emo)
    #     list = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    #     input = [[]]
    #     for item in list:
    #         input[0].append(emotMap[item])
    #     prediction = cart.predict(input)
    #     print(prediction)
    #     deleteCont('expressiveblobs')
    #     return jsonify({'like': prediction[0]}), 202

    return jsonify({'status': 'error'}), 404




if __name__ == '__main__':
    print("Listening...")
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='0.0.0.0', port=5000)

