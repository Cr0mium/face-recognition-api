import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import tqdm

from facenet_pytorch import InceptionResnetV1, MTCNN
import torch

DATA_PATH = "data/"
MODEL_PATH = "models/knn_classifier.pkl"
EMBEDDING_PATH = "embeddings/embeddings.npz"

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_images_and_labels():
    embeddings = []
    labels = []

    for person_name in tqdm(os.listdir(DATA_PATH)):
        person_path = os.path.join(DATA_PATH, person_name)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                face = mtcnn(img)
                if face is not None:
                    face_embedding = resnet(face.unsqueeze(0).to(device))
                    embeddings.append(face_embedding.detach().cpu().numpy()[0])
                    labels.append(person_name)
            except Exception as e:
                print(f"Failed processing {img_path}: {e}")

    return np.array(embeddings), np.array(labels)

def train_and_save():
    print("📦 Loading images and extracting embeddings...")
    X, y = load_images_and_labels()

    print("🏷️ Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("🤖 Training KNN classifier...")
    clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    clf.fit(X, y_encoded)

    print("💾 Saving model and embeddings...")
    joblib.dump({'model': clf, 'label_encoder': le}, MODEL_PATH)
    np.savez(EMBEDDING_PATH, X=X, y=y_encoded)

    print("✅ Training complete!")

if __name__ == "__main__":
    train_and_save()



from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

# Step 1: Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 2: Initialize MTCNN (for face detection)
mtcnn = MTCNN(image_size=160, margin=0, device=device)

# Step 3: Load FaceNet (for face embedding)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Step 4: Load a test image
img_path = "data/Chinmoy/1.jpg"  # Replace with your path
image = Image.open(img_path).convert("RGB")

# Step 5: Detect and crop face
face_tensor = mtcnn(image)
if face_tensor is None:
    print("❌ No face detected")
else:
    # Step 6: Generate embedding (1 x 512 vector)
    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0).to(device))

    print("✅ Face embedding vector shape:", embedding.shape)
    # print("🔢 First 5 values:", embedding[0][:5])
#
# 
# 
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
import io

app = FastAPI()

# Load model + encoder
model_data = joblib.load("models/knn_classifier.pkl")
clf = model_data["model"]
le = model_data["label_encoder"]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        # Load image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Detect face
        face_tensor = mtcnn(img)
        if face_tensor is None:
            return JSONResponse(content={"match": None, "confidence": 0.0, "message": "No face detected"})

        # Generate embedding
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()

        # Predict
        pred = clf.predict(embedding)[0]
        prob = max(clf.predict_proba(embedding)[0])
        name = le.inverse_transform([pred])[0]

        return JSONResponse(content={
            "match": name,
            "confidence": round(float(prob), 4)
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
 