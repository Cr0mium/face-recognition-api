import { useRef, useState } from 'react';
import React from 'react'
import Webcam from 'react-webcam'
import axios from 'axios'

function WebcamView() {
    // object to create a reference later user for screenshot
    var overlayStyle = {
        position: 'absolute',
        top: '50%',
        left: '50%',
        width: '180px',
        height: '220px',
        transform: 'translate(-50%, -50%)',
        border: '3px dashed #00ff00',
        borderRadius: '50% / 60%',
        zIndex: 10
    }
    const webcamRef = useRef(null)
    const [capturedImg, setCapturedImg] = useState(null);

    const capture = () => {
        // get screenshot and save in hook
        const imgSrc = webcamRef.current.getScreenshot();
        setCapturedImg(imgSrc)
        console.log("captured image: ", imgSrc)
    }


    const videoConstraints = {
        width: 320,
        height: 240,
        facingMode: "user", // front camera
    };

    // function cropImage(base64Image, callback) {
    //     const img = new Image();
    //     img.src = base64Image;

    //     img.onload = () => {
    //         const cropX = 30;
    //         const cropWidth = img.width - 60; // remove 30px from each side

    //         const canvas = document.createElement('canvas'); // used to draw on the browser
    //         // set the height and width of the canvas to the desired cropped image
    //         canvas.width = cropWidth;
    //         canvas.height = img.height;

    //         const ctx = canvas.getContext('2d');
    //         ctx.drawImage(img, cropX, 0, cropWidth, img.height, 0, 0, cropWidth, img.height);

    //         canvas.toBlob((blob) => {
    //             callback(blob);  // blob can be used in formData
    //         }, 'image/jpeg', 0.95);
    //     };
    // }


    const send = async () => {
        try {
            // Convert base64 to blob
            // atob-> ASCII to Binary
            const byteString = atob(capturedImg.split(',')[1]); // ['data:image/jpeg;base64', '/9j/4AAQSk...']
            //['data:image/jpeg;base64', '/9j/4AAQSk...'] -> 'image/jpeg'
            const mimeString = capturedImg.split(',')[0].split(':')[1].split(';')[0];

            const ab = new ArrayBuffer(byteString.length); // create buffer- chunk of raw memory
            const ia = new Uint8Array(ab); // view buffer as bytes- way to the memory as individual bytes

            //no fucking clue what this is doing
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }

            // final blob
            const blob = new Blob([ab], { type: mimeString });

            // Create form data
            const formData = new FormData();
            formData.append("face_image", blob, "capture.jpg");

            // Send to backend
            const response = await axios.post("http://localhost:8000/recognise", formData, {
                headers: {
                    "Content-Type": "multipart/form-data"
                }
            });

            console.log("✅ Server response:", response.data);
            alert(`Match: ${response.data.match}, Confidence: ${response.data.confidence}`);

        } catch (error) {
            console.error("❌ Failed to send image:", error);
        }
    };



    return (
        <div style={{ display: "flex", flexDirection: "column", rowGap: '20px' }}>

            <div>
                <h2>🎥 Webcam Preview</h2>
                <div style={{ position: "relative", width: 320, height: 240 }}>
                    <Webcam
                        ref={webcamRef}
                        audio={false}
                        height={240}
                        width={320}
                        mirrored={true}
                        screenshotFormat="image/jpeg"
                        videoConstraints={videoConstraints}
                    />
                    <div style={overlayStyle}></div>
                </div>
                <br />
                <button onClick={capture}>📷 Capture</button>
            </div>

            <div>
                {capturedImg && (
                    <div>
                        <div style={{ marginTop: 20 }}>
                            <h2>📸 Captured Preview:</h2>
                            <img
                                src={capturedImg}
                                alt="Captured"
                                style={{ width: "320px", height: "240px", border: "2px solid #ccc" }}
                            />
                            <br />
                            <button onClick={send}>📷 Upload to server</button>
                        </div>
                    </div>

                )}
            </div>
        </div>
    );
};

export default WebcamView;
