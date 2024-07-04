import time
import cv2 
import numpy as np
import mediapipe as mp
import threading
import pyttsx3
text_speech = pyttsx3.init()
text_speech_lock = threading.Lock() 
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

def calculate_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)

    if angle>180.0:
        angle=360-angle
    return angle



def progress_bar(progress,image):
    if progress < 0:
        progress = 0
    if progress > 1:
        progress = 1

    # Draw loading bar
    progress_bar_width = 20
    progress_bar_height = 250
    progress_bar_x = 20
    progress_bar_y = 350
    progress_bar_end_y = progress_bar_y - int(progress_bar_height * progress)

    colour=(0,255,0) #default green
    if progress>0.2:
        colour = (0, 255, 255)  # Yellow
    if progress >= 0.5:
        colour = (0, 165, 255)  # Orange
    if progress >= 0.8:
        colour = (0, 0, 255)  # Red

    cv2.rectangle(image, (progress_bar_x, progress_bar_y),
                    (progress_bar_x + progress_bar_width, progress_bar_y - progress_bar_height),
                    colour, 2) 

    cv2.rectangle(image, (progress_bar_x, progress_bar_y),
                    (progress_bar_x + progress_bar_width, progress_bar_end_y),
                    colour, -1)

    # Display progress percentage
    cv2.putText(image, f"{int(progress * 100)}%",
                (progress_bar_x, progress_bar_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    

def counter_box(image,count,stage,stage_type,excercise):
    #render counter box
    cv2.rectangle(image,(8,8),(175,90),(0,0,0),-1)

    cv2.putText(image,excercise,
                (16,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
    
    #render rep data
    cv2.putText(image,"REPS",
                (16,50),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(image,str(count),
                (18,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

    #render stage data
    if stage_type==1:
        if stage=='down':
            stage=' up'
        else:
            stage='down'
    
    cv2.putText(image,"STAGE",
                (70,50),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(image,stage,
                (65,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

def calculate_angle2(v1, v2): 
    dot_product = np.dot(v1, v2) 
    magnitude_v1 = np.linalg.norm(v1) 
    magnitude_v2 = np.linalg.norm(v2) 
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2) 
    angle = np.arccos(cos_theta) 
    return np.degrees(angle) 

def render_angle_and_line(image, landmarks): 
    back_landmark_index = 11  # For example, spine landmark index 
    hip_landmark_index = 23  # For example, left hip landmark index 
    back_point = np.array([landmarks[back_landmark_index].x * image.shape[1], landmarks[back_landmark_index].y * image.shape[0]]).astype(int) 
    hip_point = np.array([landmarks[hip_landmark_index].x * image.shape[1], landmarks[hip_landmark_index].y * image.shape[0]]).astype(int) 

    # Define a vertical line starting from the hip point 
    vertical_line_end = (hip_point[0], 0) 

    # Calculate vectors representing the back and the vertical line 
    back_vector = back_point - hip_point 
    vertical_line_vector = np.array([0, -1])  # Vertical line pointing upwards 

    # Calculate angle between the back and the vertical line 
    angle = calculate_angle2(back_vector, vertical_line_vector) 

    # Visualise angle 
    cv2.putText(image, f" {angle:.0f}", (hip_point[0] + 10, hip_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

    # Draw the vertical line 
    cv2.line(image, (hip_point[0], hip_point[1]), vertical_line_end, (255, 255, 255), 2) 
     
    return angle

class SpeechThread(threading.Thread):
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.daemon = True  # Set the thread as daemon
        self.stopped = threading.Event()  # Event to signal stop
        
    def run(self):
        global text_speech, text_speech_lock
        with text_speech_lock:
            while not self.stopped.is_set():  # Check if the thread should continue speaking
                text_speech.say(self.text)
                text_speech.runAndWait()
                
    def stop(self):
        self.stopped.set()  # Set the event to stop speaking


    
    
  
  
def alert(angle1,angle2,choice,image,count):
    engine = pyttsx3.init()
    flag=0
    colour=(0,255,0)

    if choice==0:
        down_threshold=20
        up_threshold=0
        msg1='Tuck your '
        msg2='Elbows in!'
    elif choice==1:
        down_threshold=30
        up_threshold=0
        msg1='Keep your '
        msg2='back straight!'
    elif choice==2:
        down_threshold=180
        up_threshold=160
        msg1='Keep your '
        msg2='back straight!'
    elif choice==3:
        msg1='Arms not '
        msg2='symmetric!'
    
    cv2.rectangle(image,(430,8),(630,90),colour,-1)

    if choice!=3:
        if angle2>down_threshold or angle2<up_threshold:
            flag=1
            colour=(0,0,255)
    else:
        if abs(angle1-angle2)>15:
            flag=1
            colour=(0,0,255)


    if count==0:
        cv2.putText(image,"Start your ",
                    (450,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(image,"excercise...",
                    (450,75),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2,cv2.LINE_AA)


    if count!=0 and flag==0:
        cv2.putText(image,"Good",
                    (440,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(image,"Form!",
                    (440,75),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
    elif flag==1 and count!=0:
        cv2.rectangle(image,(430,8),(630,90),colour,-1)
        cv2.putText(image,msg1,
                    (440,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(image,msg2,
                    (440,75),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2,cv2.LINE_AA)
        
        cv2.rectangle(image,(140,200),(590,270),colour,-1)
        cv2.putText(image,"INCORRECT FORM!",
                    (150,250),
                    cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),3,cv2.LINE_AA)
        # threading.Thread(target=speak, args=(msg1+msg2,)).start()
           
        speech_thread = SpeechThread(msg1 + msg2)
        
        speech_thread.start()
        speech_thread.stop()

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

