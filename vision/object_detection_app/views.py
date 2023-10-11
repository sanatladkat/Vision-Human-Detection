from django.shortcuts import render
from django.http import StreamingHttpResponse
import os
from .object_detection import camera
# ! IMPORT FEATURE MODULE HERE


# Create your views here.



def webcam_feed(request) :
    return StreamingHttpResponse(gen(camera.IpCamera()) , content_type = 'multipart/x-mixed-replace; boundary=frame')


def video_feed(request) :
    return StreamingHttpResponse(gen(camera.VideoCamera()) , content_type = 'multipart/x-mixed-replace; boundary=frame')

def gen(camera) :
    while True :
        frame = camera.get_frame()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
        )