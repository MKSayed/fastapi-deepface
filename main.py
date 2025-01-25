import base64
import multiprocessing
import os.path
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from utils import face
from fastapi.middleware.cors import CORSMiddleware

import tf_keras # For detection by pyinstaller or nuitka

SINGLE_FACE_ONLY = True
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with a specific origin if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.post('/face-verification/verify-face')
def verify_person(request: dict):
    base64_img = request["image"]
    idnum = request['idnum']
    if not base64_img or not idnum:
        return {"success": False, 'message': 'Request is missing image or idnum', "break": True}

    # Check Source Image existence
    # Source Images will always have .jpg ext
    if not os.path.exists(f'{request["idnum"]}.jpg'):
        return {'success': False, 'message': f'لا يوجد صورة مواطن برقم {idnum} في قاعدة البيانات', "break": True}

    person_face: dict | bool = False

    if SINGLE_FACE_ONLY:
        start_time = time.time()
        # retinaface is the most accurate face detection backend as of 2025-01
        faces = face.extract_faces(base64_img, 'retinaface', anti_spoofing=True, sort=False)
        end_time = time.time()
        print(f"Faces extracted in {end_time - start_time}")
        if not faces:
            person_face = False
        elif len(faces) > 1:
            return {'success': False, 'message': 'يوجد اكثر من شخص امام الكاميرا', "break": False, 'tryCount': False}
        else:
            person_face = faces[0]
    else:
        person_face = face.extract_largest_face(base64_img, 'retinaface', anti_spoofing=True)

    if not person_face:
        return {'success': False, 'message': 'برجاء وضع وجهك امام الكاميرا', "break": False, 'tryCount': False}

    print(f"Anti spoof score: {person_face['antispoof_score']}")
    is_real = person_face['is_real']
    print(is_real)

    if not is_real:
        # not not converts numpy.bool into Python bool type
        is_confident = not not (person_face['antispoof_score'] > 0.8) # Tells how confident is is_real result, lower percentage doesn't mean it's not real
        return {'success': False, 'message': f'لم يتم التعرف على المواطن - الانتحال متوقع', "break": is_confident, 'tryCount': True}

    else:
        is_confident = not not (person_face['antispoof_score'] > 0.95)
        if not is_confident:
            return {'success': False, 'message': f'اقترب قليلا', 'tryCount': False}
    # Save cropped and aligned face image
    colored_face_arr, output_folder, file_name, saved_face_img_path = face.save_face_image(person_face['face'], request['idnum'])

    # Save original captured image
    if "," in base64_img:
        base64_img = base64_img.split(",")[1]

    # Decode the Base64 string
    image_data = base64.b64decode(base64_img)

    path = Path(output_folder) / ("original - " + file_name)

    # Write the original captured image data to a file
    with open(path, "wb") as file:
        file.write(image_data)

    # ArcFace is the most accurate face recognition as of 2025-01
    start_time = time.time()
    res = face.verify(f'{idnum}.jpg', saved_face_img_path, 'ArcFace', 'retinaface',
                      enforce_detection=False, align=True, anti_spoofing=False)
    end_time = time.time()
    print(f"Faces compared in {end_time - start_time }")

    if res.get('verified', False):
        return {'success': True, 'message': 'تم المطابقة بنجاح', "break": True}
    else:
        return {'success': False, 'message': 'صورة المواطن غير مطابقة للرقم القومي المدخل', "break": False, 'tryCount': True}


if __name__=="__main__":
    multiprocessing.freeze_support()
    uvicorn.run("main:app", host='0.0.0.0')

