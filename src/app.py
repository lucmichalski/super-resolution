import os
import sys
import subprocess
import requests
import ssl
import random
import string
import json

from flask import jsonify
from flask import Flask
from flask import request
from flask import send_file

import traceback

from app_utils import blur
from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_me
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import get_multi_model_bin
from app_utils import unzip
from app_utils import unrar
from app_utils import resize_img
from app_utils import square_center_crop
from app_utils import square_center_crop
from app_utils import image_crop

import numpy as np
from PIL import Image
from ISR.models import RDN, RRDN

try:  # Python 3.5+
    from http import HTTPStatus
except ImportError:
    try:  # Python 3
        from http import client as HTTPStatus
    except ImportError:  # Python 2
        import httplib as HTTPStatus

import tensorflow as tf

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/process", methods=["POST"])
def process():

    input_path = generate_random_filename(upload_directory,"jpg")
    output_path = generate_random_filename(result_directory,"jpg")
    
    try:
        # todo: detect image type from extension
        if 'file' in request.files:
            file = request.files['file']
            if allowed_file(file.filename):
                file.save(input_path)
        else:
            url = request.json["url"]
            download(url, input_path)

        img = Image.open(input_path)
        img = img.convert('RGB')
        lr_img = np.array(img)

        # sr_img = rdn.predict(lr_img) if lr_img.shape[0] >= 360 or lr_img.shape[1] >= 640 else rrdn.predict(lr_img)
        sr_img = rrdn.predict(lr_img)
        # sr_img = rdn.predict(lr_img)
        im = Image.fromarray(sr_img)
        im.save(output_path, "JPEG")

        callback = send_file(output_path, mimetype='image/jpeg')
        return callback, 200

    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        clean_all([
            input_path,
            output_path
            ])

if __name__ == '__main__':
    global upload_directory, result_directory
    global model_directory
    global rdn
    global rrdn
    global ALLOWED_EXTENSIONS

    # use env vars
    port = os.environ.get("PORT", '5000')
    host = os.environ.get("HOST", '0.0.0.0')
    threaded = os.environ.get("THREADED", False)
    debug = os.environ.get("DEBUG", False)
    check_gpu = os.environ.get("CHECK_GPU", False)
    check_cuda = os.environ.get("CHECK_CUDA", False)

    # ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
    ALLOWED_EXTENSIONS = os.environ.get("ALLOWED_EXTENSIONS", 'png,jpg,jpeg').split(",")
    ALLOWED_EXTENSIONS = set(ALLOWED_EXTENSIONS)

    result_directory = os.environ.get("RESULTS_PATH", '/src/results/')
    create_directory(result_directory)

    upload_directory = os.environ.get("UPLOADS_PATH", '/src/uploads/')
    create_directory(upload_directory)
    
    model_directory = os.environ.get("WEIGHTS_PATH", '/src/weights/')    
    create_directory(model_directory)

    rdn = RDN(weights='noise-cancel', arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':1})
    rrdn = RRDN(weights="gans", arch_params={'C': 4, 'D': 3, 'G': 32, 'G0': 32, 'x': 1, 'T': 10})

    if check_cuda:
        print("- cuda: ", tf.test.is_built_with_cuda())

    if check_gpu:
        print("- gpu: ", tf.test.is_gpu_available())

    app.run(host=host, port=port, threaded=False, debug=debug)
