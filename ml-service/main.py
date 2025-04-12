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

#initialise models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160,margin=0,device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

#load trained weights and encoded labels
try:
    path=os.path.join(os.path.dirname(__file__),"models")
    model_data= joblib.load(f"{path}/knn_classifier.pkl")
    model=model_data['model']
    le=model_data['label_encoder']
    print(f"📊Loaded model successful {model_data}")
except Exception as e:
    print(f"❌ Failed to load Model {e}")


#like express app
app = FastAPI()

#request data structure only for json
class UserInfo(BaseModel):
    name: str
    age: int


#create get "/" route
@app.get("/")
def get_root():
    #response in the form of json
    return JSONResponse(content={"msg": f"✅test: Route working!!!"})

#create post route
@app.post("/register")
#user is the variable, userinfo is the json data structure
def get_info(user: UserInfo):
    return JSONResponse(content={"msg": f"✅test: Post working!!! User name- {user.name} age-{user.age}"})

@app.post("/recognise")
#pass the required fields as arguments
async def recognise(
    name: str =Form(...),
    age: int= Form(...),
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
        pred = model.predict(face_embedding)[0]
        prob = max(model.predict_proba(face_embedding)[0])
        name = le.inverse_transform([pred])[0]
        return JSONResponse(content={
            "match": name,
            "confidence": round(float(prob), 4)
        })
