from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, Response
from flask import render_template , request
from flask_cors import CORS, cross_origin
from imutils.video import VideoStream
from waitress import serve
import tensorflow as tf
import imutils
import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import base64
import json
import uuid
import os.path

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = './Models/facemodel.pkl'
FACENET_MODEL_PATH = './Models/20180402-114759.pb'

# Load The Custom Classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

tf.Graph().as_default()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))


# Load the model
print('Loading feature extraction model')
facenet.load_model(FACENET_MODEL_PATH)

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")
people_detected = set()
person_detected = collections.Counter()



app = Flask(__name__)

sess = tf.Session()
graph = tf.get_default_graph()
CORS(app)




@app.route('/')
@cross_origin()
def index():
    return render_template(template_name_or_list="index.html")
  #  return 'OK!';


def faceDetec(frame):

                # Phat hien khuon mat, tra ve vi tri trong bounding_boxes
                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
             #   try:
                    # Neu co it nhat 1 khuon mat trong frame
                if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # Cat phan khuon mat tim duoc
                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            
                            # Dua vao model de classifier
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]
                            
                            # Lay ra ten va ty le % cua class co ty le cao nhat
                            best_name = class_names[best_class_indices[0]]
                            #print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                            # Ve khung mau xanh quanh khuon mat
                            frame=cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 5)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            # Neu ty le nhan dang > 0.5 thi hien thi ten                    
                        
                            if best_class_probabilities > 0.5:
                                name = class_names[best_class_indices[0]]
                                frame=cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1.5, (0, 0, 0), thickness=2, lineType=2)
                                frame=cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 40),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        2, (0, 0, 0), thickness=2, lineType=2)
                                person_detected[best_name] += 1
                            else:
                                # Con neu <=0.5 thi hien thi Unknow                                
                                # Viet text len tren frame    
                                frame=cv2.putText(frame, "Unknown", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        2, (0, 0, 0), thickness=2, lineType=2)
                            
             #   except:
                #    pass

                path_file=('static/%s.jpg' %uuid.uuid4().hex)
                path_file1=('src/' + path_file)
                cv2.imwrite(path_file1,frame)               
            #    return json.dumps(path_file)
                return (path_file)

@app.route('/upload',methods=['POST'])
@cross_origin()
def upload():
    img = cv2.imdecode(np.fromstring(request.files['file'].read(),np.uint8),cv2.IMREAD_UNCHANGED)
    img_processed = faceDetec(img)
   # return Response(response=img_processed,mimetype="application/json")
    url = "localhost:5000/" + img_processed
    return render_template(template_name_or_list="Detect.html",url_link="http://localhost:5000/" + img_processed)
   # Load(url)




@app.route("/live", methods=['GET'])
@cross_origin()
def face_detect_live():
    return Response(detect_live(),mimetype='multipart/x-mixed-replace; boundary=frame')


def detect_live():
    """Detects faces in real-time via Web Camera."""

    try:
            cap = VideoStream(src=0).start()
            while (True):
                frame = cap.read()
                frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)

                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                   # if faces_found > 1:
                     #   cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          #          1, (255, 255, 255), thickness=1, lineType=2)
                if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            print(bb[i][3]-bb[i][1])
                            print(frame.shape[0])
                            print((bb[i][3]-bb[i][1])/frame.shape[0])
                            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                
                                if best_class_probabilities > 0.5:
                                    name = class_names[best_class_indices[0]]
                                else:
                                # Con neu <=0.5 thi hien thi Unknow
                                    name = "Unknown"
                                
                                # Viet text len tren frame    
                                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                                cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                                person_detected[best_name] += 1

    
                              #  cv2.imwrite('demo.jpg', frame)

                                image = cv2.imencode('.jpg',frame)[1].tobytes()
                                yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')         
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                else:
                    continue

            cap.stop()  # Stop multi-threaded Video Stream
            cv2.destroyAllWindows()
            return render_template(template_name_or_list='index.html')
    except Exception as e:
        print(e)




        

    



if __name__ == '__main__':
    serve(app=app, host='0.0.0.0', port=5000)

