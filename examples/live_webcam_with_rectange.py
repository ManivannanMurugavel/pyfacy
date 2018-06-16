from pyfacy import face_recog
from pyfacy import utils
import cv2


mdl = face_recog.Face_Recog_Algorithm()
mdl.load_model('model.pkl')

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = utils.detect_faces_locations_from_dlib(rgb)
    predictions = mdl.predict(rgb)
    print(predictions)
    # print(len(face_locations),len(predictions))
    if predictions != 0:
        for face_location,prediction in zip(face_locations, predictions):
        	top,right,bottom,left = utils.dlib_rect_to_css(face_location)
        	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    # print(utils.dlib_rect_to_css(utils.detect_faces_locations_from_dlib(rgb)[0]))
    # print(mdl.predict(rgb))

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
