import cv2
import pyvirtualcam
import numpy as np

def main():

    eye_detection_enabled = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from camera.")
        return

    frame_height, frame_width = frame.shape[:2]
    output_width, output_height = 680, 510
    desired_aspect = output_width / output_height  # For 680x510, ~1.333

    default_zoom_factor = 2.0
    default_crop_width = int(frame_width / default_zoom_factor)
    default_crop_height = int(default_crop_width / desired_aspect)

    current_crop_width = default_crop_width
    current_crop_height = default_crop_height
    current_crop_x = frame_width // 2 - current_crop_width // 2
    current_crop_y = frame_height // 2 - current_crop_height // 2

    alpha = 0.1

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    with pyvirtualcam.Camera(width=output_width, height=output_height, fps=60) as cam:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                target_face_ratio = 0.4
                dynamic_crop_width = int(w / target_face_ratio)
                dynamic_crop_height = int(h / target_face_ratio)

                if dynamic_crop_width / dynamic_crop_height < desired_aspect:
                    dynamic_crop_width = int(desired_aspect * dynamic_crop_height)
                else:
                    dynamic_crop_height = int(dynamic_crop_width / desired_aspect)

                if dynamic_crop_width < default_crop_width:
                    target_crop_width = dynamic_crop_width
                    target_crop_height = dynamic_crop_height
                else:
                    target_crop_width = default_crop_width
                    target_crop_height = default_crop_height

                target_crop_x = face_center_x - target_crop_width // 2
                target_crop_y = face_center_y - target_crop_height // 2
            else:
                target_crop_width = default_crop_width
                target_crop_height = default_crop_height
                target_crop_x = frame_width // 2 - target_crop_width // 2
                target_crop_y = frame_height // 2 - target_crop_height // 2

            current_crop_x = (1 - alpha) * current_crop_x + alpha * target_crop_x
            current_crop_y = (1 - alpha) * current_crop_y + alpha * target_crop_y
            current_crop_width = (1 - alpha) * current_crop_width + alpha * target_crop_width
            current_crop_height = (1 - alpha) * current_crop_height + alpha * target_crop_height

            current_crop_x = max(0, min(current_crop_x, frame_width - current_crop_width))
            current_crop_y = max(0, min(current_crop_y, frame_height - current_crop_height))

            cx, cy = int(current_crop_x), int(current_crop_y)
            cw, ch = int(current_crop_width), int(current_crop_height)
            cropped_frame = frame[cy:cy+ch, cx:cx+cw]
            frame_resized = cv2.resize(cropped_frame, (output_width, output_height))

            if eye_detection_enabled:
                gray_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(gray_resized, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            
                threshold_value = 50
                fill_color = (0, 255, 255)
                fill_alpha = 0.5
                
                for (ex, ey, ew, eh) in eyes:
                    eye_roi = frame_resized[ey:ey+eh, ex:ex+ew]
                    gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray_eye, threshold_value, 255, cv2.THRESH_BINARY_INV)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    fill_overlay = np.full(eye_roi.shape, fill_color, dtype=np.uint8)
                    mask_3ch = cv2.merge([mask, mask, mask])
                    mask_norm = mask_3ch.astype(float) / 255.0
                    eye_roi = (eye_roi.astype(float) * (1 - fill_alpha * mask_norm) +
                            fill_overlay.astype(float) * (fill_alpha * mask_norm))
                    eye_roi = np.clip(eye_roi, 0, 255).astype(np.uint8)
                    frame_resized[ey:ey+eh, ex:ex+ew] = eye_roi

            output_frame = frame_resized[:, :, [2, 1, 0]]

            cam.send(output_frame)
            cam.sleep_until_next_frame()

            cv2.imshow("Smooth Face Tracking with Eye Fill", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
