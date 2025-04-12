import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

# Train the model for future predictions
def train():
    #load embeddings and labels
    path=os.path.join(os.path.dirname(__file__),"embeddings")
    data = np.load(f"{path}/face_data.npz")
    embeddings = data["X"]
    labels = data["y"]
    print(f"📊 Loaded {len(embeddings)} embeddings for {len(np.unique(labels))} people.")
    # encoding the names to numbers [john,doe]->[0,1]
    le = LabelEncoder()
    encoded_labels=le.fit_transform(labels)

    # Train the model on the numpy arrays
    clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')#Create
    clf.fit(embeddings, encoded_labels)

    #quick check to see the accuracy of the model on the training data itself
    accuracy = clf.score(embeddings, encoded_labels)
    print(f"✅ Training accuracy: {accuracy:.2%}")#<variable>:.<decimal places>%<*100>}
    
    # Save it
    # joblib.dump(clf, 'models/knn_classifier.pkl')) #for single parameter
    try:
        path=os.path.join(os.path.dirname(__file__),"models")
        os.makedirs(path,exist_ok=True)
        
        #used for general purpose python object
        joblib.dump({     
            "model":clf,
            "label_encoder": le
        }, f"{path}/knn_classifier.pkl")
        print("📂 knn_classifier pkl saved")
    except Exception as e:
        print(f"❌ Failed to save pkl: {e}")
    
if __name__ == "__main__":
    train()