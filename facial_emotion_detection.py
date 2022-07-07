
import cv2
from keras.models import load_model
from fer import FER
import tensorflow as tf

face = cv2.CascadeClassifier(
    'haar cascade files\haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# using pre-trained FER model
emotion_detector = FER()
# using your own trained model from train.py
# emotion_detector = tf.keras.models.load_model('models/model.h5')

while (True):
    ret, input_image = cap.read()

    result = emotion_detector.detect_emotions(input_image)
    if result != []:
        bounding_box = result[0]["box"]
        emotions = result[0]["emotions"]
        cv2.rectangle(input_image, (
            bounding_box[0], bounding_box[1]), (
            bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
            (100, 155, 255), 2,)

    score_max = 0
    for key in emotions.items():
        if key[1] > score_max:
            score_max = key[1]

        for index, (emotion_name, score) in enumerate(emotions.items()):

            color = (0, 0, 255) if score == score_max else (255, 0, 0)
            emotion_score = "{}: {}".format(
                emotion_name, "{:.2f}".format(score))

            cv2.putText(input_image, emotion_score,
                        (20, 20 +
                         + index * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,)

    cv2.imshow('a', input_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
