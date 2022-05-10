import cv2
import uuid

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret,frame = cap.read()
    imgname = "{}.jpg".format(str(uuid.uuid1()))
    cv2.imwrite(f"without mask\{imgname}",frame)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()