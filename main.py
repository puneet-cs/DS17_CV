# Computer Vision based Face Filter Project

# Library used : cv2, mediapipe

import cv2   # used to capture video
import mediapipe as mp  # to identify the key points on face
import numpy as np

# We are using transparent images for filter

# Load the images
# CV2 image channels : B G R A 
mustache_png = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)
hat_png = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)

# checking if image is transparent or not
for name, img in (("mustache.png",mustache_png ), ("hat.png", hat_png)):
    if img is None or img.shape[2] < 4:
        raise FileNotFoundError(f"Missing or invaild picture {name}")

# identify the landmark on the face
mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh(
    static_image_mode = False,   # detect the face in video
    max_num_faces = 1,           # maximum 1 face will detect
    refine_landmarks = True,      # low resolution images
    min_detection_confidence = 0.5,  # consider it as a face if prob >= 0.5
    min_tracking_confidence = 0.5  # if Prob > 0.5, keep tracking face
)

# overlay the mustache and hat on the frame
def overlay_rgba(background, overlay, x, y, w, h ):
    overlay = cv2.resize(overlay, (w,h), interpolation = cv2.INTER_AREA)

    b, g, r, a = cv2.split(overlay)

    alpha = a.astype(float) / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    # get background frame size
    h_bg, w_bg = background.shape[:2]

    # overlay don't fo outside the frame
    x0, y0 = max(0,x), max(0,y) # top left cornver of the overlay
    x1, y1 = min(x + w, w_bg), min( y + h, h_bg)   # Bottom right corner

    # define the region of the overlay image  when the some part of overlay go outside the frame
    overlay_slice = (slice(y0-y, y1 - y), slice(x0 - x, x1 - x))
    background_roi = (slice(y0, y1), slice(x0, x1))

    # Extract the region of interest
    forground = cv2.merge([b,g,r])[overlay_slice]
    alpha_roi = alpha[overlay_slice]
    bg_roi = background[background_roi]

    # blending of overlay with background
    blended = cv2.convertScaleAbs(forground  * alpha_roi + bg_roi * (1 - alpha_roi) )
    background[background_roi] = blended
    return background



# Using the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Cannot capture video")

# process real time webcam frames 
while True:
    ok, frame = cap.read()

    if not ok:
        print("Empty frame is captured")
        break

    frame = cv2.flip(frame, 1)  # flip the image horizontally, like selfie
    h_frame, w_frame = frame.shape[:2]  

    # Converting BGR to RGB , mediapipe library use RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get the landmark on captured frame
    result = face_mesh.process(rgb)

    # get the landmarks on the first detected face
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # convert the landmark into pixels
        def to_px(idx):
            pt = landmarks[idx]
            return int(pt.x * w_frame ), int(pt.y * h_frame)
        
        # find the pixel position for mustache
        # upperlip landmark index in mediapipe = 13
        # lowerlip landmark index  in mediapipe = 14
        lip_x1, lip_y1 = to_px(13)
        lip_x2, lip_y2 = to_px(14)

        lip_x = (lip_x1 + lip_x2) // 2
        lip_y = (lip_y1 + lip_y2) // 2

        # find the pixel for placing the hat
        left_temple_x, _ = to_px(127)  # 127: left temple landmark index
        right_temple_x, _ = to_px(356)  # 356 : right temple landmark
        forehead_x, forehead_y = to_px(10) # 10: forhead center landmark index

        face_w = right_temple_x - left_temple_x  # face width

        # adding mustache on background frame
        must_w = face_w
        must_h = int(must_w * 0.30)
        must_x = lip_x - must_w // 2
        must_y = lip_y - int(must_h * 0.75)
        frame = overlay_rgba(frame, mustache_png, must_x, must_y , must_w, must_h )

        # Add Hat on the background frame
        hat_w = int(face_w * 1.6)
        hat_h = int(hat_w * 0.9)
        hat_x = forehead_x - hat_w // 2
        hat_y = forehead_y - int(hat_h * 0.8)
        frame = overlay_rgba(frame, hat_png, hat_x, hat_y , hat_w, hat_h )
    
    cv2.imshow("Press ESC to end the video : press s to capture image", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite("selfie.png", frame)
        print("Image saved successfully")

cap.release()
cv2.destroyAllWindows()