import cv2 
from markdown import Markdown
import numpy as np
import mediapipe as mp
import pyttsx3
from mlp_functions2 import calculate_angle, progress_bar, counter_box, alert, render_angle_and_line
import threading
import pathlib
import textwrap

import google.generativeai as genai



def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def gym(choice):

    text_speech = pyttsx3.init()
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    GOOGLE_API_KEY="AIzaSyCSc7mz3nIPxI7eLCfTOlEiUyRnXoGic7Q"

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')




    # captures the video
    cap = cv2.VideoCapture(0)

    # counter and stage variables
    count = 0
    stage = None

    if choice == 0:
        first = 11
        second = 13
        third = 15
        down_threshold = 160
        up_threshold = 30
        stage_type = 0
        exercise = 'Curls'
        fourth = 23
        fifth = 11
        sixth = 13
    elif choice == 1:
        first = 23
        second = 25
        third = 27
        down_threshold = 160
        up_threshold = 80
        stage_type = 1
        exercise = 'Squats'
    elif choice == 2:
        first = 11
        second = 13
        third = 15
        down_threshold = 160
        up_threshold = 90
        stage_type = 1
        exercise = 'Push-ups'
        fourth = 27
        fifth = 23
        sixth = 11
    elif choice == 3:
        first = 11
        second = 13
        third = 15
        down_threshold = 160
        up_threshold = 80
        stage_type = 1
        exercise = 'Shoulder press'
        fourth = 12
        fifth = 14
        sixth = 16
    
    elif choice==4:
        exit()
    else:
        print('invalid')
        exit()
    # setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()  # takes in the frames

            # recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # make detection
            results = pose.process(image)

            # recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # extract landmarks
            try:
                if results.pose_landmarks is not None:
                    landmarks = results.pose_landmarks.landmark

                    # get coordinates
                    a = [landmarks[first].x, landmarks[first].y]
                    b = [landmarks[second].x, landmarks[second].y]
                    c = [landmarks[third].x, landmarks[third].y]

                    # calculate angle
                    angle1 = calculate_angle(a, b, c)
                    angle1 = int(round(angle1, 3))

                    # visualise
                    cv2.putText(image, str(angle1),
                                tuple(np.multiply(b, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                    if choice != 1:
                        # get coordinates for form correction
                        d = [landmarks[fourth].x, landmarks[fourth].y]
                        e = [landmarks[fifth].x, landmarks[fifth].y]
                        f = [landmarks[sixth].x, landmarks[sixth].y]

                        # calculate angle
                        angle2 = calculate_angle(d, e, f)
                        angle2 = int(round(angle2))

                        # visualise
                        cv2.putText(image, str(angle2),
                                    tuple(np.multiply(e, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                    else:
                        angle2 = render_angle_and_line(image, landmarks)

                    alert(angle1, angle2, choice, image, count)


                    # counter
                    if angle1 > down_threshold:
                        stage = 'down'
                    if angle1 < up_threshold and stage == 'down':
                        stage = ' up'
                        count += 1

                    # Calculate the progress of the loading bar based on the angle
                    progress = 1 - ((angle1 - up_threshold) / (down_threshold - up_threshold))  # Normalize the angle between 0 and 1
                    progress_bar(progress, image)  # -------- RENDERS LOADING BAR

            except Exception as e:
                print(e)

            counter_box(image, count, stage, stage_type, exercise)  # ------ RENDERS COUNTER BOX

            # render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                    )

            cv2.imshow('Mediapipe Feed', image)  # shows the video
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                # to exit press 'q'
                #break
                #do this to return count to function call
                return count       
            