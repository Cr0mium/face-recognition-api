import os
from tqdm import tqdm
from PIL import Image
import torch 
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

DATA_PATH = os.path.join(os.path.dirname(__file__),"data")

#check if gpu(cuda) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model to detect face using facenet, cropped to 160*160
mtcnn = MTCNN(image_size=160,margin=0,device=device)
#Runs the FaceNet model and returns a 512-d embedding that uniquely represents that face
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

#load images to preprocess and generate embeddings 
def load_data():
    if not os.path.exists(DATA_PATH) or not os.path.isdir(DATA_PATH):
        print("Path does not exists")
        return [],[]
    
    if not os.listdir(DATA_PATH):
        print("⚠️ No data found in 'data/' directory.")
        return [], []

    embeddings=[]
    labels=[]
    skipped=0
    for person_name in tqdm(os.listdir(DATA_PATH),desc="🔍 Processing people"):
        if not os.path.isdir(os.path.join(DATA_PATH, person_name)): 
                continue
        for image in tqdm(os.listdir(os.path.join(DATA_PATH,person_name)),desc=f"📷 {person_name}",leave=False):
            image_path=os.path.join(DATA_PATH,person_name,image)
            try:
                #open the image using pillow, convert to RGB for mtcnn
                img = Image.open(image_path).convert("RGB")
                #process image detection
                face_tensor = mtcnn(img)
                if face_tensor is not None:
                    #extract embeddings
                    face_embedding = resnet(face_tensor.unsqueeze(0).to(device))
                    #add the embedding in the list, detach the tensor which was in gpu, 
                    #shift to cpu for numpy, pick the 1st element of a vector [1,512]->[512]
                    embeddings.append(face_embedding.detach().cpu().numpy()[0])
                    labels.append(person_name)
                else:
                    skipped+=1
            except Exception as e:
                print(f"Failed process {image_path}: {e}")

    print(f"🚫 Skipped {skipped} images with no detectable face.")
    # convert to numpy array to use on KNC, which expects numpy as input
    return np.array(embeddings),np.array(labels)


if __name__ == "__main__":
    embeddings,labels = load_data()
    print("✅ Done! Loaded embeddings shape:", embeddings.shape)
    # print("🧾 Sample labels:", np.unique(names))
    try:
        path=os.path.join(os.path.dirname(__file__),"embeddings")
        os.makedirs(path, exist_ok=True)

        
        data_to_save= {
            "X": embeddings,
            "y": labels,
        }
        #specialised for numpy arrays, faster
        # np.savez(f"{path}/face_data.npz", X=embeddings, y=labels)
        np.savez(f"{path}/face_data.npz",**data_to_save)
        print(f"💾 Saved embeddings/face_data.npz")
        print(f"📊 Saved {len(embeddings)} embeddings for {len(np.unique(labels))} people.")
    except Exception as e:
        print(f"❌ Failed to save npz: {e}")
