import cv2
from ultralytics import YOLO
import face_recognition as fr
import numpy as np
import torch
import time
import traceback
import os
import pickle
from dotenv import load_dotenv
from datetime import datetime
from twilio.rest import Client
from defs import DEFAULT_SEND_SMS_ON_NEW_UNKNOWN
from defs import DEFAULT_SEND_SMS_ON_RETURNING_UNKNOWN 
from defs import DEFAULT_SMS_COOLDOWN_SECONDS
import ultralytics.nn.autobackend as ab
import types

# Load environment variables
load_dotenv()

# ROCm: disable fusion globally
if torch.version.hip is not None:
    def no_fuse(self, verbose=False):
        # Simply return self without fusing
        return self
    ab.AutoBackend.fuse = no_fuse  # patch AutoBackend.fuse
    
# Twilio configuration from .env file
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER")

# SMS Settings (can be set in .env or use defaults)
SEND_SMS_ON_NEW_UNKNOWN = os.getenv("SEND_SMS_ON_NEW_UNKNOWN", DEFAULT_SEND_SMS_ON_NEW_UNKNOWN if isinstance(DEFAULT_SEND_SMS_ON_NEW_UNKNOWN, bool) else True).lower() == "true"
SEND_SMS_ON_RETURNING_UNKNOWN = os.getenv("SEND_SMS_ON_RETURNING_UNKNOWN", DEFAULT_SEND_SMS_ON_NEW_UNKNOWN if isinstance(DEFAULT_SEND_SMS_ON_RETURNING_UNKNOWN, bool) else False).lower() == "true"
SMS_COOLDOWN_SECONDS = int(os.getenv("SMS_COOLDOWN_SECONDS", DEFAULT_SMS_COOLDOWN_SECONDS if isinstance(DEFAULT_SEND_SMS_ON_RETURNING_UNKNOWN, str) else "300"))

# Check if Twilio is properly configured
TWILIO_ENABLED = all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER])
if not TWILIO_ENABLED:
    print("Warning: Twilio not configured. Please set TWILIO_* variables in .env file. SMS alerts disabled.")

def load_known_faces(faces_directory: str):
    """
    Load all known faces from a directory.
    Each image file (jpg/png) becomes a known person, with filename as their name.
    
    Args:
        faces_directory: Path to directory containing face images
        
    Returns:
        Tuple of (encodings_list, names_list)
    """
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(faces_directory):
        print(f"Warning: Faces directory '{faces_directory}' not found. Creating it...")
        os.makedirs(faces_directory, exist_ok=True)
        print(f"Please add face images (jpg/png) to {faces_directory}")
        return known_face_encodings, known_face_names
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Scan directory for image files
    image_files = []
    for filename in os.listdir(faces_directory):
        file_path = os.path.join(faces_directory, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append((filename, file_path))
    
    if not image_files:
        print(f"Warning: No face images found in '{faces_directory}'")
        print(f"Add .jpg or .png files with person names (e.g., John.jpg, Sarah.png)")
        return known_face_encodings, known_face_names
    
    # Load each face image
    print(f"Loading known faces from '{faces_directory}'...")
    for filename, file_path in image_files:
        try:
            # Extract person name from filename (without extension)
            person_name = os.path.splitext(filename)[0]
            
            # Load and encode the face
            image = fr.load_image_file(file_path)
            encodings = fr.face_encodings(image)
            
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
                print(f"Loaded: {person_name} ({filename})")
            else:
                print(f"No face detected in: {filename}")
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"Total known faces loaded: {len(known_face_encodings)}")
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


def send_sms_alert(message: str, last_sms_time: float) -> float:
    """
    Send SMS alert via Twilio with cooldown protection.
    Returns the timestamp of when SMS was sent (or last_sms_time if not sent).
    """
    if not TWILIO_ENABLED:
        return last_sms_time
    
    try:
        # Check cooldown
        current_time = time.time()
        if current_time - last_sms_time < SMS_COOLDOWN_SECONDS:
            print(f"SMS cooldown active. Next SMS allowed in {int(SMS_COOLDOWN_SECONDS - (current_time - last_sms_time))}s")
            return last_sms_time
        
        # Send SMS
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message_obj = client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=TWILIO_TO_NUMBER
        )
        print(f"✓ SMS sent successfully (SID: {message_obj.sid})")
        return current_time
        
    except Exception as e:
        print(f"Error sending SMS: {e}")
        print(traceback.format_exc())
        return last_sms_time


def main():
    # Set environment variable to help debug ROCm errors
    os.environ["AMD_SERIALIZE_KERNEL"] = "3"
    
    # Load YOLOv8 model with fusing disabled for ROCm compatibility
    os.environ["YOLO_DISABLE_FUSE"] = "1"
    model = YOLO("yolov8s.pt", verbose=False)
    
    # Force CPU usage (GPU disabled due to compatibility issues)
    use_gpu = False
    print("Using CPU")
    model.to("cpu")
    
    # Disable half precision on ROCm (causes issues with AMD GPUs)
    yolo_half = False    
    # Load all known faces from faces/ directory (relative to script location)
    faces_directory = os.path.join(os.path.dirname(__file__), "faces")
    print(f"Loading known faces from {faces_directory}")
    known_face_encodings, known_face_names = load_known_faces(faces_directory)

    # Open webcam
    video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Throttled recognition logger
    last_log_time = 0.0
    pending_names = set()
    last_sms_time = 0.0  # Track last SMS sent time for cooldown

    # Unknown face database (persistent across runs)
    unknown_db_path = os.path.join(os.path.dirname(__file__), "unknown_faces.pkl")
    if os.path.exists(unknown_db_path):
        try:
            with open(unknown_db_path, "rb") as f:
                unknown_db = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load unknown faces database: {e}")
            unknown_db = []
    else:
        unknown_db = []

    def save_unknown_db():
        """Save the unknown faces database to disk"""
        try:
            with open(unknown_db_path, "wb") as f:
                pickle.dump(unknown_db, f)
        except Exception as e:
            print(f"Error saving unknown faces database: {e}")

    def next_unknown_id() -> str:
        """Generate next available unknown ID"""
        existing = [int(entry["id"].split("_")[-1]) for entry in unknown_db 
                   if entry.get("id", "").startswith("unk_") and entry["id"].split("_")[-1].isdigit()]
        return f"unk_{(max(existing) + 1) if existing else 1}"

    # Threshold for considering two unknown encodings the same person (lower = stricter)
    unknown_match_tolerance = 0.7

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # YOLO object detection (prefer GPU if available; lower imgsz; allow half precision)
        yolo_results = model(
            frame,
            device=0 if use_gpu else "cpu",
            imgsz=640,
            half=yolo_half,
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
        unknown_events = []  # Track new/returning unknown faces for logging
        
        for face_encoding in face_encodings:
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                # Known face recognized
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                # Unknown face - check against database
                if unknown_db:
                    stored_encodings = [entry["encoding"] for entry in unknown_db]
                    distances = fr.face_distance(stored_encodings, face_encoding)
                    best_idx = int(np.argmin(distances))
                    best_dist = float(distances[best_idx])
                else:
                    best_idx = -1
                    best_dist = 1.0

                now_ts = time.time()
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if best_idx >= 0 and best_dist <= unknown_match_tolerance:
                    # Previously seen unknown person - update their record
                    entry = unknown_db[best_idx]
                    entry["last_seen_ts"] = now_ts
                    entry["last_seen"] = now_str
                    entry["seen_count"] = entry.get("seen_count", 0) + 1
                    name = entry["id"]
                    unknown_events.append((name, now_str, "returning", entry.get("first_seen", "unknown")))
                else:
                    # New unknown person - create record
                    uid = next_unknown_id()
                    unknown_db.append({
                        "id": uid,
                        "encoding": face_encoding,
                        "first_seen_ts": now_ts,
                        "first_seen": now_str,
                        "last_seen_ts": now_ts,
                        "last_seen": now_str,
                        "seen_count": 1,
                    })
                    name = uid
                    unknown_events.append((uid, now_str, "new", None))
                    save_unknown_db()  # Save immediately for new entries
                    
            face_names.append(name)

        # Collect recognized names for throttled logging
        for name in face_names:
            if name != "Unknown":
                pending_names.add(name)

        # Print at most every 0.5s if any names recognized or unknown faces detected
        now = time.time()
        if now - last_log_time >= 0.5:
            if pending_names:
                print("Recognized:", ", ".join(sorted(pending_names)))
                pending_names.clear()
            
            # Log unknown face events with details and send SMS alerts
            if unknown_events:
                for uid, timestamp, event_type, first_seen in unknown_events:
                    if event_type == "new":
                        print(f"NEW UNKNOWN: {uid} first detected at {timestamp}")
                        
                        # Send SMS alert for new unknown person
                        if SEND_SMS_ON_NEW_UNKNOWN:
                            sms_message = f"ALERT: Unknown person ({uid}) detected in your room at {timestamp}"
                            last_sms_time = send_sms_alert(sms_message, last_sms_time)
                            
                    else:
                        print(f"RETURNING: {uid} seen again at {timestamp} (first seen: {first_seen})")
                        
                        # Optionally send SMS for returning unknowns
                        if SEND_SMS_ON_RETURNING_UNKNOWN:
                            sms_message = f"ALERT: Unknown person ({uid}) returned at {timestamp}"
                            last_sms_time = send_sms_alert(sms_message, last_sms_time)
                            
                # Batch save after processing all events in this interval
                save_unknown_db()
            
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


