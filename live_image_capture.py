import cv2
import mediapipe as mp
import math
import numpy as np

# Constants for landmarks
FOREHEAD_LANDMARK = 10
CHIN_LANDMARK = 152
DEPTH_THRESHOLD = 200  # Adjusted after checking the values of depth
# Eye landmark indices
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3
LONG_CLOSURE_FRAMES = 25

# State variables
closed_eye_frames = 0
eye_was_closed = False
blink_count = 0

capture_flag = False
image_captured = False
label = ""

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,max_num_faces=2)

# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640) #(property_id for width,width)
cap.set(4, 480) #(property_id for height,height)

# Calculate eye aspect ratio
def calculate_ear(eye):      
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])#horizontal distance
    return (A + B) / (2.0 * C)

# Calculate Euclidean distance
def euclidean_distance(pt1, pt2):  
    return math.dist(pt1,pt2)
face_states={}
while True:
    suc,img = cap.read()
    if not suc:
        break
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    cx, cy = w // 2, h // 2
    radius = 150
    colors = (0, 0, 255)

    if result.multi_face_landmarks:
        if len(result.multi_face_landmarks) > 1:
            cv2.putText(img, "MULTIPLE FACES DETECTED", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        for face_id, face_landmarks in enumerate(result.multi_face_landmarks):#facelandmark is an obj for each face
            landmarks=face_landmarks.landmark # to get landmarks from each face
            if face_id not in face_states:#face_states is a dictionary where each face ID stores its blink state and counters.
                face_states[face_id] = {'closed_eye_frames': 0, 'eye_was_closed': False, 'blink_count': 0}
            state = face_states[face_id]
           
            # Face center
            face_x = int(landmarks[1].x * w)#to get actual position
            face_y = int(landmarks[1].y * h)
            dist_to_center = math.dist((face_x, face_y), (cx, cy))
            #print(dist_to_center)
            if dist_to_center < radius - 80:
                colors = (0, 255, 0)  # Aligned
           
            # Get forehead and chin coordinates
            forehead = (int(landmarks[FOREHEAD_LANDMARK].x * w), int(landmarks[FOREHEAD_LANDMARK].y * h))
            chin = (int(landmarks[CHIN_LANDMARK].x * w), int(landmarks[CHIN_LANDMARK].y * h))
            # Calculate distance
            depth = euclidean_distance(forehead, chin)
            #print("face depth:",depth)
           
            # Eye coordinates
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_IDX]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_IDX]
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2
            #print("eye aspect ratio:",avg_ear)

            # Eye state
            if avg_ear < EAR_THRESHOLD:
                state['closed_eye_frames'] += 1
                if state['closed_eye_frames'] >= CONSEC_FRAMES:
                    state['eye_was_closed'] = True        
            else:
                if state['eye_was_closed']:
                    state['blink_count'] += 1
                    state['eye_was_closed'] = False
                    if state['closed_eye_frames'] > LONG_CLOSURE_FRAMES and not image_captured and dist_to_center < radius - 50 and avg_ear < 0.26:
                        capture_flag = True
                        image_captured=True
                state['closed_eye_frames'] = 0
           
            # Determine fake or real
            if depth > DEPTH_THRESHOLD and state['blink_count'] >=1 :
                label = "Real Person"
                color = (0, 255, 0)  
            else:
                label = "Fake Photo"
                color = (0, 0, 255)

            # Draw bounding box
            x_coords = [int(lm.x * w) for lm in landmarks]
            y_coords = [int(lm.y * h) for lm in landmarks]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

           # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(img, f"{label}", (x_min, y_min-70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # Draw face alignment circle
    cv2.circle(img, (cx, cy), radius, colors, 3)

    if capture_flag :
    # Create a circular mask
        mask = np.zeros_like(img)
        cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1)
    # Apply the mask to the image
        masked_img = cv2.bitwise_and(img, mask)
    # Crop square region around the circle
        x1 = max(cx - radius, 0)
        y1 = max(cy - radius, 0)
        x2 = min(cx + radius, img.shape[1])
        y2 = min(cy + radius, img.shape[0])
        cropped_img = masked_img[y1:y2, x1:x2]
    # Save the cropped image
        cv2.imwrite("image.jpg", cropped_img)
    # Show the cropped result
        cv2.imshow("Captured Face", cropped_img)
        cv2.waitKey(4000)
        break
   
    cv2.imshow("Live Image Capture", img)
    # To exit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()












