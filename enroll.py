import cv2, os

def enroll():
    roll = input("Enter student roll number (e.g. RM001): ").strip()
    name = input("Enter student name: ").strip()
    folder = f"enrolled_db/{roll}_{name.replace(' ','_')}"
    os.makedirs(folder, exist_ok=True)
    cam = cv2.VideoCapture(0)
    count = 0
    print("Webcam open. Press SPACE to take photo (take 3). Press Q to quit.")
    while count < 3:
        ret, frame = cam.read()
        cv2.imshow(f"Enrolling {name}", frame)
        key = cv2.waitKey(1)
        if key == ord(' '):
            cv2.imwrite(f"{folder}/photo{count+1}.jpg", frame)
            count += 1
            print(f"Photo {count} saved!")
        elif key == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    print(f"Done! {name} enrolled.")

if __name__ == "__main__":
    while True:
        enroll()
        if input("Enroll another? (yes/no): ").strip().lower() != "yes":
            break