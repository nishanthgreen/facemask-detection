import cv2
import numpy as np
from tensorflow.python.keras.models import load_model

model = load_model('face_150.hdf5')

cap = cv2.VideoCapture(0)

label = ["with_mask", "without_mask"]
color = [(0, 255, 0), (0, 0, 255)]


def detect_and_predict_mask(frame, faceNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    locs = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            locs.append((startX, startY, endX, endY))

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return locs


prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

while True:
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)

    locs = detect_and_predict_mask(frame, faceNet)
    try:
        for i in locs:
            (startX, startY, endX, endY) = i

            #cv2.imshow("face", frame[startY:endY, startX:endX])
            gray = cv2.cvtColor(frame[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY)
            data = cv2.resize(gray, (100, 100))
            data = np.array(data) / 255.0
            data = np.reshape(data, (1, 100, 100, 1))

            #print(model.predict(data))

            prediction = np.argmax(model.predict(data))

            cv2.rectangle(frame, (startX, startY), (endX, endY), color[prediction], 2)
            cv2.putText(frame, label[prediction], (startX, startY-10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (color[prediction]), 2)
            cv2.putText(frame, "fps:{}".format(str(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0) , 2)
    except:
        cv2.putText(frame, "Face not recognized", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
