import face_recognition
import numpy as np
import math
import cv2
import os, sys


def face_confidence(face_dist, face_match_threshold=0.6):
    range = 1.0 - face_match_threshold
    lin_val = (1.0 - face_dist) / (range * 2.0)
    
    if face_dist > face_match_threshold:
        return str(round(lin_val*100, 2)) + "%"
    else:
        val = (lin_val + ((1.0 - lin_val) * math.pow((lin_val - 0.5) * 2, 0.2))) * 100
        return str(round(val, 2)) + "%"
        

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
        
    def encode_faces(self):
        path = 'people'
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        for img_path in image_paths:
            face_img = face_recognition.load_image_file(img_path)
            face_encoding = face_recognition.face_encodings(face_img)[0]

            self.known_face_encodings.append(face_encoding)
            
            # Extract the base filename from the path (remove the directory)
            img_filename = os.path.basename(img_path)
            self.known_face_names.append(img_filename)


    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)


        while True:
            ret, frame = video_capture.read()
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                # rgb_small_frame = small_frame[:,:,::-1]
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "Unknown"

                    face_dist = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_idx = np.argmin(face_dist)

                    if matches[best_match_idx]:
                        name = self.known_face_names[best_match_idx]
                        confidence = face_confidence(face_dist[best_match_idx])

                    self.face_names.append(f"{name} ({confidence})")

            self.process_current_frame = not self.process_current_frame

            for (top,right,bottom,left), name in zip(self.face_locations, self.face_names):
                top *= 4
                bottom *= 4
                right *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()



fr = FaceRecognition()
fr.run_recognition()