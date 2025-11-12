import cv2
from ultralytics import YOLO
import face_recognition as fr
import numpy as np
import torch
import time


def load_known_face(known_image_path: str):
    known_face_encodings = []
    known_face_names = []

    image = fr.load_image_file(known_image_path)
    encodings = fr.face_encodings(image)
    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        print(encodings[0])
        known_face_names.append("Ahmed")
    return known_face_encodings, known_face_names


def draw_yolo_detections(frame, results):
    # Use Ultralytics built-in plot for convenience
    # This returns an annotated frame with boxes and labels
    return results[0].plot()


def draw_face_recognition(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
    return frame


def main():
    # Load YOLOv8 model (expects yolov8s.pt in repo root)
    model = YOLO("yolov8s.pt")
    # If PyTorch is built with ROCm on AMD, torch.cuda will be available and map to HIP
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.to("cuda")

    # Load known face (Ahmed)
    known_image_path = "/home/ahmed/Downloads/ahmed.png"
    known_face_encodings, known_face_names = load_known_face(known_image_path)

    # Open webcam
    video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Throttled recognition logger
    last_log_time = 0.0
    pending_names = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # YOLO object detection (prefer GPU if available; lower imgsz; allow half precision)
        yolo_results = model(
            frame,
            device=0 if use_gpu else "cpu",
            imgsz=640,
            half=use_gpu,
            verbose=False,
        )
        annotated = draw_yolo_detections(frame.copy(), yolo_results)

        # Face recognition (process downscaled frame on CPU; rescale boxes back up)
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        scale = 0.5
        small_rgb = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale)
        face_locations_small = fr.face_locations(small_rgb, model="hog")
        try:
            face_encodings = fr.face_encodings(small_rgb, face_locations_small, num_jitters=0, model="small")
        except TypeError:
            face_encodings = fr.face_encodings(small_rgb, num_jitters=0, model="small")

        # Upscale locations to original coordinates
        face_locations = []
        for (top, right, bottom, left) in face_locations_small:
            face_locations.append((int(top / scale), int(right / scale), int(bottom / scale), int(left / scale)))

        face_names = []
        for face_encoding in face_encodings:
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

        # Collect recognized names for throttled logging
        for name in face_names:
            if name != "Unknown":
                pending_names.add(name)

        # Print at most every 0.5s if any names recognized
        now = time.time()
        if now - last_log_time >= 0.5:
            if pending_names:
                print("Recognized:", ", ".join(sorted(pending_names)))
                pending_names.clear()
            else:
                print("No names recognized")
            last_log_time = now

        # Draw face boxes and labels on top of YOLO annotations
        annotated = draw_face_recognition(annotated, face_locations, face_names)

        cv2.imshow("Objects + Face Recognition", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


