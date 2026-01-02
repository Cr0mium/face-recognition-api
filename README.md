# 🧠 face-recognition-api

A full-stack, real-time face recognition system for attendance and identity verification. Combines deep learning (FaceNet embeddings with PyTorch) and classical ML (KNN classifier) in a scalable, Docker-first architecture.

---

## ✨ Features

- 🎯 Face detection with MTCNN
- 🧠 Face embedding with InceptionResnetV1 (FaceNet)
- 🧪 Classification with scikit-learn’s KNN
- 📷 React frontend with live webcam capture
- 🎨 Custom face-frame overlay for alignment
- 🧾 Registration with multiple image support
- 🔁 Model auto-update on new user addition
- 🗃️ Embeddings & models saved with NumPy + joblib
- 🐳 Docker-ready architecture (multi-service support)

---

## 🧰 Tech Stack

- **Backend**: Python · FastAPI · PyTorch · scikit-learn · joblib
- **Frontend**: React.js · Vite · Axios · react-webcam
- **Infra**: Docker · Docker Compose · REST API

---

## 🗂️ Project Structure

```bash
.
├── ml-service/           # FastAPI + Face Recognition API
├── frontend/             # React frontend (webcam interface)
├── backend/              # Optional Node.js server (DB/API extension)
├── embeddings/           # Saved face embeddings
├── models/               # Trained KNN classifier
├── data/                 # Uploaded face images (organized by name)
├── uploads/              # Recognized face logs
├── postgres/             # DB init scripts (future)
└── docker-compose.yml
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/face-recognition-api.git
cd face-recognition-api
```

### 2. Start the services
```bash
docker-compose up --build
```

### 3. Access the frontend
Visit `http://localhost:5173`

---

## 📦 API Endpoints

### `POST /register`
Registers a new person with multiple face images.

- `Form`: name (str)
- `File[]`: images

### `POST /recognise`
Recognizes a face from a single image.

- `File`: face_image (UploadFile)

Returns:
```json
{
  "match": "Angelina_Jolie",
  "confidence": 0.92
}
```

---

## 🛣️ Roadmap

- [x] Face detection & recognition API
- [x] Dynamic registration & retraining
- [x] React webcam UI with live preview
- [ ] PostgreSQL integration for attendance logs
- [ ] JWT-based admin & user authentication
- [ ] Cloud deployment with CI/CD

---

## 📸 Demo

> Coming soon – screenshots, video preview, and hosted app link.

---

## 📄 License

MIT License © Chinmoy Deka
