import cv2
import os

# === CONFIG ===
person_name = "Chinmoy_Deka"  # Replace or make input()
output_dir = os.path.join("data", person_name)
os.makedirs(output_dir, exist_ok=True)

img_width, img_height = 320, 240
max_images = 5
delay_between_captures = 1000  # milliseconds

# === INIT CAMERA ===
cap = cv2.VideoCapture(0)
cap.set(3, img_width)
cap.set(4, img_height)

print(f"📸 Starting webcam for {person_name}. Press SPACE to capture images.")

count = 0
while cap.isOpened() and count < max_images:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)  # 👈 Flip horizontally (mirror effect)

    # Display the frame
    cv2.imshow("Register Face - Press SPACE", frame)

    # Wait for key press
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == 32:  # SPACE to capture
        filename = os.path.join(output_dir, f"{person_name}_{count+1}.jpg")
        resized = cv2.resize(frame, (img_width, img_height))
        cv2.imwrite(filename, resized)
        print(f"✅ Saved {filename}")
        count += 1
        cv2.waitKey(delay_between_captures)  # optional delay


# Release everything
cap.release()
cv2.destroyAllWindows()
print("👋 Done. Images saved to:", output_dir)
