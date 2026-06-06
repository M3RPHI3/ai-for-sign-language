import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def main(camera=0):
    cap = cv2.VideoCapture(camera)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        success, img = cap.read()

        if not success:
            print("Cannot read frame")
            break

        img = cv2.flip(img, 1)

        # แปลงเป็น RGB ก่อนส่งเข้า MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hand_result = hands.process(rgb)
        pose_result = pose.process(rgb)

        # วาด Pose
        if pose_result.pose_landmarks:
            mp_draw.draw_landmarks(
                img,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # วาด Hands
        if hand_result.multi_hand_landmarks:
            for hand_landmark in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("MediaPipe", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()