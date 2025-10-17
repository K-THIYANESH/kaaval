import os
import cv2

DATASET_DIR = 'dataset'

person = input('Enter person name (no spaces, e.g. John_Doe): ').strip()
if not person:
    print('Invalid name')
    raise SystemExit(1)

person_dir = os.path.join(DATASET_DIR, person)
os.makedirs(person_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press 's' to save a face snapshot, 'q' to quit.")
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Capture - press s to save', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        path = os.path.join(person_dir, f'{person}_{count}.jpg')
        cv2.imwrite(path, frame)
        print('Saved', path)
        count += 1
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Done.')
