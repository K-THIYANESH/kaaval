import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Compact, efficient recognizer using facenet-pytorch embeddings
# Dataset layout: dataset/<PersonName>/*.jpg
DATASET_DIR = 'dataset'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(keep_all=True, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)


def build_database(dataset_dir=DATASET_DIR):
    encodings = []
    names = []
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    for person in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in os.listdir(person_dir):
            path = os.path.join(person_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # mtcnn will crop and align face(s)
            boxes, _ = mtcnn.detect(img_rgb)
            if boxes is None:
                continue
            try:
                face_img = mtcnn.extract(img_rgb, boxes, None)[0]
            except Exception:
                continue
            face_tensor = face_img.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = resnet(face_tensor).cpu().numpy()[0]
            encodings.append(embedding)
            names.append(person)
    if encodings:
        return np.array(encodings), names
    return np.empty((0,512)), []


def find_best_match(embedding, db_encodings, db_names, threshold=0.8):
    if db_encodings.size == 0:
        return None, None
    # cosine similarity
    emb = embedding / np.linalg.norm(embedding)
    dbn = db_encodings / np.linalg.norm(db_encodings, axis=1, keepdims=True)
    sims = np.dot(dbn, emb)
    idx = np.argmax(sims)
    return db_names[idx], sims[idx]


def recognize_from_frame(frame, db_encodings, db_names):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img_rgb)
    if boxes is None:
        return []
    faces = mtcnn.extract(img_rgb, boxes, None)
    results = []
    for face in faces:
        face_tensor = face.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = resnet(face_tensor).cpu().numpy()[0]
        name, score = find_best_match(emb, db_encodings, db_names)
        results.append((name, float(score)))
    return results


def live_mode(db_encodings, db_names):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = recognize_from_frame(frame, db_encodings, db_names)
        # draw results
        for i, (name, score) in enumerate(res):
            label = (name if name is not None else 'Unknown') + f':{score:.2f}'
            cv2.putText(frame, label, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def identify_image(path, db_encodings, db_names):
    img = cv2.imread(path)
    if img is None:
        print('Failed to load image')
        return
    res = recognize_from_frame(img, db_encodings, db_names)
    print(res)


def main():
    print('Building database...')
    db_encodings, db_names = build_database()
    print('DB size:', len(db_names))
    print('Choose option:\n1. Live camera\n2. Identify image')
    c = input('Enter 1 or 2: ')
    if c == '1':
        live_mode(db_encodings, db_names)
    elif c == '2':
        p = input('Image path: ')
        identify_image(p, db_encodings, db_names)
    else:
        print('Invalid')

if __name__ == '__main__':
    main()