import datetime
import os
import traceback
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

# build model once
model = DeepFace.build_model("ArcFace")

model2 = DeepFace.build_model("retinaface", task='face_detector')

def verify(
    img1_path: str,
    img2_path: str,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    """
    :param img1_path: Can Either be a path to the image with any extension or data:image/jpeg;base64,<base64 data> string
    :param img2_path: Can Either be a path to the image with any extension or data:image/jpeg;base64,<base64 data> string
    :param model_name:
    :param detector_backend:
    :param enforce_detection:
    :param align:
    :param anti_spoofing:
    """
    try:
        res = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return res
    except ValueError as err:
        if traceback.format_exc().find('Spoof'):
            # print(traceback.format_exc())
            return {"verified": False, "spoof": True}
        else:
            raise err
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"verified": False}



def analyze(
    img_path: str,
    actions: list,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
            anti_spoofing=anti_spoofing,
        )
        result["results"] = demographies
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400


def extract_faces(img_path: str, detector_backend: str, anti_spoofing: bool, sort=False):
    try:
        # Extract faces with alignment
        face_objs = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            align=True,
            enforce_detection=True, # Will raise an exception if no faces where detected
            anti_spoofing = anti_spoofing,
        )

        if sort:
            # Sort faces by face surface area in the image
            face_objs = sorted(face_objs, key=lambda x:
            x['facial_area']['w'] * x['facial_area']['h']  # Use width(w) and height(h) keys
                               )
        return face_objs
    except ValueError as e:
        print(f"No faces detected: {str(e)}")
        return False

def extract_largest_face(img_path: str, detector_backend: str, anti_spoofing: bool):
    try:
        # Extract faces with alignment
        face_objs = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            align=True,
            enforce_detection=True, # Will raise an exception if no faces where detected
            anti_spoofing=anti_spoofing
        )
        # Find the face with the largest area
        largest_face = max(face_objs, key=lambda x:
        x['facial_area']['w'] * x['facial_area']['h']  # Use width(w) and height(h) keys
                           )

        return largest_face

    except ValueError as e:
        print(f"No faces detected: {str(e)}")
        return False

def save_face_image(face_img, img_name):
    """
    Expects numpy.ndarray containing image data
    """
    output_file_name_suffix = str(datetime.datetime.now()).split('.')[0].replace(':', '-')

    # 1. Convert from float64 (0-1 range) to uint8 (0-255)
    if face_img.dtype == np.float64:
        face_img = (face_img * 255).astype(np.uint8)

    # 2. Convert RGB to BGR for cv2 saving
    colored_face_arr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

    file_name = f"{img_name} - {output_file_name_suffix}.jpg"
    output_folder = 'images'

    # Define the path
    path = Path(output_folder)

    # Create the directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

    path = path / file_name

    cv2.imwrite(str(path), colored_face_arr)

    print(f"Saved face to {path}")

    return colored_face_arr, output_folder, file_name, path # Either of those can be used directly in the verify method
