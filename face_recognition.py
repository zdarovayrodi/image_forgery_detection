import cv2
from deepface import DeepFace
from typing import List, Dict, Any



def detect_faces(img_path: str) -> List[Dict[str, Any]]:
    img = cv2.imread(img_path)
    detected_faces = DeepFace.extract_faces(img)
    return detected_faces


def extract_face(img_path: str, face_data: Dict[str, Any]) -> Any:
    img = cv2.imread(img_path)
    facial_area = face_data['facial_area']
    x,y,w,h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
    face_img = img[y:y+h, x:x+w]
    return face_img
    


if __name__ == '__main__':
    img = cv2.imread('img/original/1.jpg')
    detected_faces = detect_faces('img/original/1.jpg')
    
    for i, face_data in enumerate(detected_faces):
        face_img = extract_face('img/original/1.jpg', face_data)
        
        cv2.imshow(f'Extracted Face {i+1}', face_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    for face_data in detected_faces:
        face = face_data['face']
        facial_area = face_data['facial_area']
        x,y,w,h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        cv2.rectangle(img, (x, y), (x+w, y+h), (128, 255, 0), 2)
        
    cv2.imshow('Detected Faces', img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

