# import os, io;
# from google.cloud import vision;
# import pandas as pd;
#
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:/Users/prana/hackathon/Bose Home Hub-1bced939195e.json"
# client = vision.ImageAnnotatorClient()
#
# image_path = "C:/Users/prana/PycharmProjects/faceDetection/smile.png"
#
# with io.open(image_path, 'rb') as image_file:
#     content = image_file.read()
#
# image = vision.types.Image(content=content)
# response = client.face_detection(image=image)
# faceAnnotations = response.face_annotations
#
# likehood = ('Unknown', 'Very Unlikely', 'Unlikely', 'Possibly', 'Likely', 'Very Likely')
#
# print('Faces:')
# for face in faceAnnotations:
#     print('Detection Confidence {0}'.format(face.detection_confidence))
#     print('Angry likelyhood: {0}'.format(likehood[face.anger_likelihood]))
#     print('Joy likelyhood: {0}'.format(likehood[face.joy_likelihood]))
#     print('Sorrow likelyhood: {0}'.format(likehood[face.sorrow_likelihood]))
#     print('Surprised ikelihood: {0}'.format(likehood[face.surprise_likelihood]))
#     print('Headwear likelyhood: {0}'.format(likehood[face.headwear_likelihood]))
#
#     face_vertices = ['({0},{1})'.format(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
#     print('Face bound: {0}'.format(', '.join(face_vertices)))
#     print('')


# import os, io;
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:/Users/prana/hackathon/Bose Home Hub-1bced939195e.json"
#
# def detect_faces_uri(uri):
#     """Detects faces in the file located in Google Cloud Storage or the web."""
#     from google.cloud import vision
#     client = vision.ImageAnnotatorClient()
#     image = vision.types.Image()
#     image.source.image_uri = uri
#
#     response = client.face_detection(image=image)
#     faces = response.face_annotations
#
#     # Names of likelihood from google.cloud.vision.enums
#     likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
#                        'LIKELY', 'VERY_LIKELY')
#     print('Faces:')
#
#     for face in faces:
#         print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
#         print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
#         print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
#
#         vertices = (['({},{})'.format(vertex.x, vertex.y)
#                     for vertex in face.bounding_poly.vertices])
#
#         print('face bounds: {}'.format(','.join(vertices)))
#
#
#
# if __name__ == '__main__':
#     detect_faces_uri("gs://sfaf/smile.png")


# import os;
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:/Users/prana/hackathon/Bose Home Hub-1bced939195e.json"
#
# def detect_faces(path):
#     """Detects faces in an image."""
#     from google.cloud import vision
#     import io
#     client = vision.ImageAnnotatorClient()
#
#     with io.open(path, 'rb') as image_file:
#         content = image_file.read()
#
#     image = vision.types.Image(content=content)
#
#     response = client.face_detection(image=image)
#     faces = response.face_annotations
#
#     # Names of likelihood from google.cloud.vision.enums
#     likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
#                        'LIKELY', 'VERY_LIKELY')
#     print('Faces:')
#
#     for face in faces:
#         print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
#         print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
#         print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
#         print('sadness: {}'.format(likelihood_name[face.sorrow_likelihood]))
#
#
#         vertices = (['({},{})'.format(vertex.x, vertex.y)
#                     for vertex in face.bounding_poly.vertices])
#
#         print('face bounds: {}'.format(','.join(vertices)))
#
# detect_faces("C:/Users/prana/Flask API/sanket.jpg")


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


def detect_faces_url(url):
    import requests
    import json

    # set to your own subscription key value
    subscription_key = "759d10fd86c74db19b7c791f66b90752"
    assert subscription_key

    # replace <My Endpoint String> with the string from your endpoint URL
    face_api_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'

    image_url = url
    # image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Puma_face.jpg/320px-Puma_face.jpg'

    headers = {'Ocp-Apim-Subscription-Key': subscription_key,
               "Content-Type": "application/json"}

    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'emotion',
    }

    response = requests.post(face_api_url, params=params,
                             headers=headers, json={"url": image_url})
    json_data = response.json() if response and response.status_code == 200 else None
    if json_data == None or len(json_data) == 0:
        return None
    print(json_data[0]['faceAttributes']['emotion'])
    return json_data[0]['faceAttributes']['emotion']


def computeAverage(urls):
    import json;

    map = {}

    emotions = {'anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'}
    for emot in emotions:
        map[emot] = 0

    counter = 0
    for url in urls:
        emotions = detect_faces_url(url)
        print(emotions)
        if (emotions == None):
            continue
        counter += 1
        for emotion in emotions:
            if emotions[emotion] > map[emotion]:
                map[emotion] = emotions[emotion]

            # value = emotions[emotion]
            # map[emotion] += value

    # maxEmot = float("-inf")
    # for key, value in map.items():
    #     if value > maxEmot:
    #         map[key] = value


    return map


def determineEmotion(map):
    if len(map.items()) == 0:
        return None
    max = float("-inf")
    maxKey = "something"
    for key, value in map.items():
        if map[key] > max and key != 'neutral':
            max = map[key]
            maxKey = key
    if (maxKey == 'anger' or maxKey == 'contempt' or maxKey == 'disgust' or maxKey == 'fear' or maxKey == 'sadness'):
        if (max > 0.2):
            return 1
        return 0
    else:
        return 1






# Main method.
if __name__ == '__main__':
    # createContainer('expressionblobs3')
    # addBlob('smile.png', 'expressionblobs3')
    #
    # addBlob('sanket.jpg', 'expressionblobs3')
    # addBlob('excitedface.png', 'expressionblobs3')
    # addBlob('sadface.png', 'expressionblobs3')
    # urls = urlList('expressionblobs3')
    # emotMap = computeAverage(urls)
    # print(emotMap)
    # print(determineEmotion(emotMap))
    cart = trainDataset()
    #detect_faces_url('https://bosehomehub.blob.core.windows.net/expressionblobs4/sm.jpg')
    urls = urlList('expressionblobs4')
    emotMap = computeAverage(urls)
    list = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    input = [[]]
    for item in list:
        input[0].append(emotMap[item])
    prediction = cart.predict(input)
    print(prediction)




