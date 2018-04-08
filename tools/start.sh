#!/bin/bash

export FLASK_APP=image_recognition_http.py
python -m flask run 

# test example:
# time curl -v  -F file=@data/demo/baidu_logo.jpg http://127.0.0.1:5000/upload

#curl -i -X POST  -H "Content-Type:application/json"  -H "Accept:application/json;charset=utf-8" \
#   -d '{ "app_id":"10116606", "session_id":"1522736589560162", "image":"/9j/4AAQSkZJRgAB...."}' \
# 'http://127.0.0.1:5000/imageRecognition'