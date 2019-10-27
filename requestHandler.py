from flask import Flask, request, abort, jsonify

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
        return jsonify({'status': 'ok'}), 201

    if request.method == 'GET':
        print("helllo")
        urls = urlList('expressiveblobs')
        emotMap = computeAverage(urls)
        print(emotMap)
        emo = determineEmotion(emotMap)
        print(emo)
        deleteCont('expressiveblobs')
        return jsonify({'emotion': emo}), 202

    return jsonify({'status': 'error'}), 404




if __name__ == '__main__':
    print("Listening...")
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='0.0.0.0', port=5000)

