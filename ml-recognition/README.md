Kaaval - Face recognition starter

Usage

1. Activate Python 3.10 virtualenv (.venv310):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { . .venv310\Scripts\Activate }"
```

2. Populate the dataset. Use the helper to capture images:

```powershell
python add_face_from_webcam.py
# follow prompt to enter name, press 's' to save snapshots
```

3. Run the recognizer:

```powershell
python kaaval.py
```

Notes
- Dataset layout: `dataset/<PersonName>/*.jpg`
- For damaged or partial faces, image preprocessing and advanced inpainting models are required; this starter does basic recognition using facenet-pytorch (MTCNN + InceptionResnetV1).
- If you need help installing Python 3.10 or setting up the environment, ask and I'll guide you step-by-step.
