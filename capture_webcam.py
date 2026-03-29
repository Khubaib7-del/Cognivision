import cv2
import os
import time

def capture_image(save_path="data/raw_images"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")

    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened. Press 's' to save an image or 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame.")
            break

        # Display the resulting frame
        cv2.imshow('Cognivision - Capture Image', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Save the image with a timestamp
            filename = f"classroom_{int(time.time())}.jpg"
            file_path = os.path.join(save_path, filename)
            cv2.imwrite(file_path, frame)
            print(f"Image saved to: {file_path}")
        
        elif key == ord('q'):
            # Quit the loop
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
