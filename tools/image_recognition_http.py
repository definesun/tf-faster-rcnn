import image_recognition
import json
import base64
import cv2
from flask import Flask
from flask import request
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/imageRecognition', methods=['POST'])
def image_recog():
    print(request.json)
        
    img_base64 = request.json['image']
    img = base64.b64decode(img_base64)
    bb=image_recognition.ir.img_rec(img)
    return json.dumps({'result': bb})


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('/tmp/tmp.jpg')
        img = cv2.imread('/tmp/tmp.jpg')
        bb=image_recognition.ir.img_rec(img)

        return json.dumps({'result': bb})
    return 'Hello, World!'

