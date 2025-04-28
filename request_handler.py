from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import UUID
import base64
import logging
import os
import cv2 as cv
import numpy as np
import time
import asyncio
import threading

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODEL_DIR = "Libs"
PROTO_FILE_NAME = "pose_deploy_linevec.prototxt"
MODEL_FILE_NAME = "pose_iter_440000.caffemodel"

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
# combination of pose pairs
POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

PROTO_PATH = os.path.join(BASE_DIR, MODEL_DIR, PROTO_FILE_NAME)
MODEL_PATH = os.path.join(BASE_DIR, MODEL_DIR, MODEL_FILE_NAME)

net = cv.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

logging.info("Model loaded successfully")

# warm up the model with a dummy input
dummy_blob = cv.dnn.blobFromImage(np.zeros((368, 368, 3), dtype=np.uint8), 1.0/255, (368, 368))
net.setInput(dummy_blob)
_ = net.forward()

net_lock = threading.Lock()

app = FastAPI()

class Image(BaseModel):
    id: str
    image: str

class Detector():
    def __init__(self, net,
                 img_width = 368,
                 img_height = 368,
                 threshold = 0.3):
        self.net = net
        self.img_width = img_width
        self.img_height = img_height
        self.threshold = threshold

    def detect_keypoints(self, image):
        # net = cv.dnn.readNetFromCaffe(self.proto, self.model)
        t0 = time.time()
        image_bytes = self.base64_to_image(image.image)
        img = self.decode_image(image_bytes)

        # logging.debug("Image shape, %s", str(img.shape))

        width = img.shape[1]
        height = img.shape[0]

        
        inp = cv.dnn.blobFromImage(img, 1.0 / 255, (self.img_width, self.img_height), (0, 0, 0), swapRB=False, crop=False)
        t1 = time.time()

        out = None
        with net_lock:
            self.net.setInput(inp)
            out = self.net.forward()
        
        if out is None:
            raise Exception("Failed to get output from the model")
        
        t2 = time.time()

        points = []
        results = []
        min_x, min_y, max_x, max_y = width, height, 0, 0

        # Don't count background in the results.
        for i in range(len(BODY_PARTS) - 1):
            heatmap = out[0, i, :, :]

            _, conf, _, point = cv.minMaxLoc(heatmap)
            x = (width * point[0]) / out.shape[3]
            y = (height * point[1]) / out.shape[2]

            if conf > self.threshold:
                points.append((int(x), int(y)))
                results.append((int(x), int(y),float(conf)))
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        box = None
        if points:
            box = {
                "x": int(min_x),
                "y": int(min_y),
                "width": int(max_x - min_x),
                "height": int(max_y - min_y),
                "probability": 1.0
            }

        t3 = time.time()

        detected_result = {
            "id": image.id,
            "count": 1 if points else 0,
            "boxes": [box] if box else [],
            "keypoints": [results] if points else [],
            "speed_preprocess": round((t1 - t0) * 1000, 4),
            "speed_inference": round((t2 - t1) * 1000, 4),
            "speed_postprocess": round((t3 - t2) * 1000, 4)
        }

        return detected_result

    def draw_keypoints(self, image):
        image_bytes = self.base64_to_image(image.image)
        img = self.decode_image(image_bytes)
        
        width = img.shape[1]
        height = img.shape[0]

        inp = cv.dnn.blobFromImage(img, 1.0 / 255, (self.img_width, self.img_height), (0, 0, 0), swapRB=False, crop=False)
        
        out = None
        with net_lock:
            self.net.setInput(inp)
            out = net.forward()
        
        if out is None:
            raise Exception("Failed to get output from the model")

        points = []
        results = []
        for i in range(len(BODY_PARTS)):
            heatmap = out[0, i, :, :]

            _, conf, _, point = cv.minMaxLoc(heatmap)
            x = (width * point[0]) / out.shape[3]
            y = (height * point[1]) / out.shape[2]

            points.append((int(x), int(y)) if conf > self.threshold else None)
            results.append((int(x), int(y),float(conf)) if conf > self.threshold else None)

        for pair in POSE_PAIRS:
            partfrom = pair[0]
            partto = pair[1]
            assert(partfrom in BODY_PARTS)
            assert(partto in BODY_PARTS)

            id_from = BODY_PARTS[partfrom]
            id_to = BODY_PARTS[partto]
            if points[id_from] and points[id_to]:
                cv.line(img, points[id_from], points[id_to], (255, 74, 0), 3)
                cv.ellipse(img, points[id_from], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
                cv.ellipse(img, points[id_to], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
                cv.putText(img, str(id_from), points[id_from], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)
                cv.putText(img, str(id_to), points[id_to], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)

        success, encoded_image = cv.imencode('.jpg', img)
        if not success:
            raise Exception("Failed to encode image")

        image_bytes = encoded_image.tobytes()
        encoded_image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return {
            "id": image.id,
            "image": encoded_image_base64
        }

    def decode_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        # logging.info("Image loaded from RAW bytes successfully")
        return img

    def base64_to_image(self, base64_string):
        image_bytes = base64.b64decode(base64_string)
        # logging.info("Image decoded from base64 successfully")
        return image_bytes

    def test_function(self):
        pass

detector = Detector(net)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/keypoints")
async def keypoints(image: Image):
    try:
        # logging.info("Received image with id: %s", image.id)
        # logging.info("Image string: %s", image.image)
    
        response = await asyncio.to_thread(detector.detect_keypoints, image)
        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}")
    
@app.post("/annotated")
async def annotated_image(image: Image):
    try:
        # logging.info("Received image with id: %s", image.id)
        # logging.info("Image string: %s", image.image)
        
        response = await asyncio.to_thread(detector.draw_keypoints, image)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}")


