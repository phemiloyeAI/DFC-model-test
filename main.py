import os
import cv2
import time
import json
import tempfile
import numpy as np

from utils import choose_weights, DFC_inference, get_dfc_crop_label, extract_HSV_mask, CropCoverageArea
from matplotlib import pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()

def download_file(name_file: str, media_type):
    return FileResponse(path=os.getcwd() + "/" + name_file, media_type=media_type, filename=name_file)


@app.get("/")
def home():
    return {"Inference on DFC Model"}
    
@app.post("/uploadfile/{crop_stage}")
def create_upload_file(crop_stage: int, file: UploadFile=File(..., )):
    start = time.time()

    tfile = tempfile.NamedTemporaryFile(delete=False)

    tfile.write(file.file.read())

    fname = tfile.name

    # load image
    image_size = (224, 224)
    inputImg = cv2.imread(fname)

    fname = file.filename.split(".")[0]
    inputImg = cv2.resize(inputImg, image_size, interpolation = cv2.INTER_AREA)

    info = {}

    info["Filename"] = fname
    info["Crop Stage"] = crop_stage
    
    # choose weight   

    weight_path = choose_weights( crop_stage, "weights/")
    weights = weight_path.split("/")[-1]
    info["Weights"] = weights

    print(f"[INFO] model loaded for stage {crop_stage}")

    # get DFC mask
    dfc_output = DFC_inference(inputImg, weight_path)
    print("[INFO] generated DFC mask ")

    # get HSV mask
    hsv_mask = extract_HSV_mask(inputImg) 
    print("[INFO] extracted HSV mask ")

    # map DFC and HSV masks
    crop_pixel, crop_color = get_dfc_crop_label( dfc_output, hsv_mask ) 
    print("[INFO] Found the crop label ")

    # calculate crop coverage area
    crop_area = CropCoverageArea( crop_pixel, dfc_output)

    # crop coverage area
    info["CropCoverageArea"] = str(crop_area) + " %"
    print(f"[INFO] {fname} has Crop Coverage Area of {crop_area}%")

    input_viz = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)
    output = np.concatenate((input_viz, dfc_output, hsv_mask, crop_color), axis = 1)
    dst =  fname + "_output.png"
    info["Output dst"] = dst

    cv2.imwrite(dst, output)
    with open("info.json", "w") as fp:
         json.dump(info, fp, indent=2)

    
    # download to disk
#     download_file(name_file=dst, media_type="image/jpeg") #segmentation image
#     download_file(name_file="info.json", media_type="info/json") #info json file
    
    end = time.time()
    duration = round(end-start, 3)
    print(f"It takes {duration} seconds to make the prediction.")

    return info
