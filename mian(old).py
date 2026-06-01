import mediapipe as mp
import cv2, os, time, math

hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
frame_x, frame_y = 720, 480

def main(camera):
    cap = cv2.VideoCapture(camera)
    
    while True:
        ret, img = cap.read()
        if not ret:
            print("video end")
            break
        img = cv2.flip(img, 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(img)
        pose_result = pose.process(img)

        main_point = None

        row=[]
        
        if pose_result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img,pose_result.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS)
            # for id, lm in enumerate(pose_result.pose_landmarks.landmark):
                # cx, cy = int(lm.x * frame_x), int(lm.y * frame_y)
                # cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if hand_result.multi_hand_landmarks:
            for idx, hand_landmark in enumerate(hand_result.multi_hand_landmarks):
                mp.solutions.drawing_utils.draw_landmarks(img,hand_landmark,mp.solutions.hands.HAND_CONNECTIONS)
      
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main(0)