import cv2
import os
import time

def main():
    # Directory setup
    base_dir = "data/raw"
    labels = ["attentive", "distracted"]
    
    for label in labels:
        path = os.path.join(base_dir, label)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created folder: {path}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- CogniVision Data Collector ---")
    print("Instructions:")
    print("Press 'a' - Save as ATTENTIVE")
    print("Press 'd' - Save as DISTRACTED")
    print("Press 'q' - Quit collector")
    print("----------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display instructions on screen
        display_frame = frame.copy()
        cv2.putText(display_frame, "A: Attentive | D: Distracted | Q: Quit", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("CogniVision Collector", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            filename = f"attent_{int(time.time())}.jpg"
            cv2.imwrite(os.path.join(base_dir, "attentive", filename), frame)
            print(f"Saved: {filename} to attentive/")
            
        elif key == ord('d'):
            filename = f"distract_{int(time.time())}.jpg"
            cv2.imwrite(os.path.join(base_dir, "distracted", filename), frame)
            print(f"Saved: {filename} to distracted/")
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
