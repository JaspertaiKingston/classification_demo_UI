import seaborn as sns
import matplotlib.pyplot as plt
import requests
import io
from PIL import Image
import numpy as np
import cv2

def post(uploaded_file, n, url="http://127.0.0.1:8080/image-classification"):
    params = {"numberofpred": n}
    res = requests.post(
        url,
        params=params,
        files={"image": uploaded_file.getvalue()}
        )
    return res.json()

def read_img(uploaded_file, resize=True):
    imagefile = io.BytesIO(uploaded_file.read())
    im = Image.open(imagefile)
    if resize:
        im_array = np.array(im.resize((224,224)))
    else:
        im_array = np.array(im)
    return im_array

def plot_result(uploaded_file, results):
    prediction = {results[result]['label']:results[result]['prob'] for result in results}
    img_plot = read_img(uploaded_file)

    fig = plt.figure(figsize = (9,4))
    ax = fig.add_axes([0,0,1,1])
    ax_sub = fig.add_axes([0.4,0.15,0.8,0.8])
    ax.bar(prediction.keys(), prediction.values(), color='#2CBDFE')
    ax_sub.imshow(img_plot)
    ax_sub.axis('off')
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    return fig

def post_azure(uploaded_file, url="http://127.0.0.1:8080/image_caption"):
    res = requests.post(
        url,
        files={"image": uploaded_file.getvalue()}
        )
    return res.json()

def plot_object(uploaded_file, info = None):
    image = Image.open(uploaded_file)
    image = np.array(image)

    if info is not None:
        objects = info['Object']

        for i in range(len(objects)):
            loc = objects[i]['boundingBox']
            top_left = (loc['x'], loc['y']+loc['h'])
            bottom_right = (loc['x']+loc['w'],loc['y'])

            color = (0, 255, 0)
            thickness = 2
            image = cv2.rectangle(image, top_left, bottom_right, color, thickness)

            font_type = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.7
            font_color = (0, 255, 0)
            image = cv2.putText(
                image, 
                f"{objects[i]['tags'][0]['name']} ({objects[i]['tags'][0]['confidence']:.2f})", 
                (loc['x'], loc['y']-4), 
                font_type, font_size, font_color, 2
            )
    return image

