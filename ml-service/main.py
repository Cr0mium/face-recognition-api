import os
from fastapi.responses import JSONResponse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from fastapi import FastAPI,UploadFile,File,Form
from pydantic import BaseModel
from PIL import Image
import torch 
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import io
from typing import List
from trainer import train
from datetime import datetime

#initialise models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160,margin=0,device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

#load trained weights and encoded labels
def load_knn():
    try:
        path=os.path.join(os.path.dirname(__file__),"models")
        model_data= joblib.load(f"{path}/knn_classifier.pkl")
        model=model_data['model']
        le=model_data['label_encoder']
        print(f"📊Loaded model successful {model_data}")
        return model,le
    except Exception as e:
        print(f"❌ Failed to load Model {e}")


#like express app
app = FastAPI()

#request data structure only for json
class UserInfo(BaseModel):
    name: str
    age: int

def update_model(new_embeddings, new_labels):
#load embeddings and labels to add new face
    path=os.path.join(os.path.dirname(__file__),"embeddings")
    data = np.load(f"{path}/face_data.npz")
    embeddings = data["X"]
    labels = data["y"]
    #cannot directly append numpy arrays
    embeddings = np.concatenate([embeddings, new_embeddings], axis=0)
    labels = np.concatenate([labels, new_labels], axis=0)


    try:
        path=os.path.join(os.path.dirname(__file__),"embeddings")
        os.makedirs(path, exist_ok=True)

        #specialised for numpy arrays, faster
        data_to_save= {
            "X": embeddings,
            "y": labels,
        }
        # np.savez(f"{path}/face_data.npz", X=embeddings, y=labels)
        np.savez(f"{path}/face_data.npz",**data_to_save)
        print(f"💾 Saved embeddings/face_data.npz")
        print(f"📊 Saved {len(embeddings)} embeddings for {len(np.unique(labels))} people.")
        train()
    except Exception as e:
        print(f"❌ Failed to save npz: {e}")
    
    
#create get "/" route
@app.get("/")
def get_root():
    #response in the form of json
    return JSONResponse(content={"msg": f"✅test: Route working!!!"})

#create post route
@app.post("/register")
#user is the variable, userinfo is the json data structure
async def register(
    name: str= Form(...),
    images: List[UploadFile] = File(...)
    ):
    new_embeddings = []
    skipped=0
    #format the label
    new_labels= ["_".join(name.split())]*len(images)
    for i,file in enumerate(images):
        try:
            img_bytes = await file.read()
            image= Image.open(io.BytesIO(img_bytes)).convert("RGB")
            face_tensor= mtcnn(image)
            if face_tensor is not None:
                face_embedding = resnet(face_tensor.unsqueeze(0).to(device))
                #add the embedding in the list, detach the tensor which was in gpu, 
                #shift to cpu for numpy, pick the 1st element of a vector [1,512]->[512]
                new_embeddings.append(face_embedding.detach().cpu().numpy()[0])
                path= os.path.join(os.path.dirname(__file__),"data","_".join(name.split()))
                os.makedirs(path, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image.save(os.path.join(path, f"{timestamp}_{i+1}.jpg"))
            else:
                skipped+=1
        except Exception as e:
            print(f"Failed process {file.filename}: {e}")

    if(skipped):
            return JSONResponse(content={"msg": f"{skipped} images were skipped, Adding additional images will help with accuracy"})
    else:
         update_model(new_embeddings,new_labels)

        
    return JSONResponse(content={"label": new_labels})
        
    

@app.post("/recognise")
#pass the required fields as arguments
async def recognise(
    face_image: UploadFile = File(...)
):
    #load image
    img_bytes= await face_image.read() #fastapi reads the file as raw bytes
    #raw bytes is converted into image using io, which in opened as normal image
    image= Image.open(io.BytesIO(img_bytes)).convert("RGB")
    #detect face tensor
    face_tensor= mtcnn(image)
    if face_tensor is None:
            return JSONResponse(content={"match": None, "confidence": 0.0, "message": "No face detected"})
    else:
        #if face is detected, generated embeddings
        #pytorch by default keep tracks of gradient (gradient decent-packpropagation)
        #disable gradient tracking
        with torch.no_grad():
            face_embedding= resnet(face_tensor.unsqueeze(0).to(device)).detach().cpu().numpy()

        # predict
        # for prediction .numpy()[0] is not used as the expected vector in [1,512]
        # for saving embeddings, we use [512]
        model,le=load_knn()
        pred = model.predict(face_embedding)[0]
        prob = max(model.predict_proba(face_embedding)[0])
        name = le.inverse_transform([pred])[0]
        if prob < 0.6:
            return {"match": "Unknown", "confidence": prob}
        path= os.path.join(os.path.dirname(__file__),"uploads")
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image.save(os.path.join(path, f"{name}_{prob:%}_{timestamp}.jpg"))
        return JSONResponse(content={
            "match": name,
            "confidence": round(float(prob), 4)
        })