import cv2, os, pandas as pd
from deepface import DeepFace
from datetime import datetime

def run():
    batch = input("Enter Batch ID (e.g. RM-B04): ").strip()
    cam = cv2.VideoCapture(0)
    print("Webcam open. Press SPACE to take group photo.")
    photo_path = None
    while True:
        ret, frame = cam.read()
        cv2.imshow("Group Photo - press SPACE", frame)
        if cv2.waitKey(1) == ord(' '):
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            photo_path = f"group_photos/{batch}_{ts}.jpg"
            cv2.imwrite(photo_path, frame)
            print("Group photo saved!")
            break
    cam.release()
    cv2.destroyAllWindows()

    print("Recognising faces... please wait.")
    candidates = []
    for folder in os.listdir("enrolled_db"):
        parts = folder.split("_", 1)
        if len(parts) == 2:
            candidates.append({"roll": parts[0], "name": parts[1].replace("_", " ")})

    matched = set()
    try:
        faces = DeepFace.extract_faces(photo_path, enforce_detection=False)
        for i, face in enumerate(faces):
            import numpy as np
            face_img = (face["face"] * 255).astype("uint8")
            tmp = f"_tmp_{i}.jpg"
            cv2.imwrite(tmp, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            try:
                result = DeepFace.find(tmp, db_path="enrolled_db",
                                       model_name="Facenet512",
                                       enforce_detection=False, silent=True)
                if result and len(result[0]) > 0:
                    dist = result[0].iloc[0]["distance"]
                    if dist < 0.4:
                        folder = os.path.normpath(result[0].iloc[0]["identity"]).split(os.sep)[-2]
                        matched.add(folder.split("_")[0])
            except: pass
            if os.path.exists(tmp): os.remove(tmp)
    except Exception as e:
        print(f"Face detection error: {e}")

    records = [{"Roll No": c["roll"], "Name": c["name"],
                "Status": "Present" if c["roll"] in matched else "Absent",
                "Date": datetime.now().strftime("%d-%m-%Y"),
                "Batch": batch} for c in candidates]

    df = pd.DataFrame(records)
    out = f"attendance_records/{batch}_{datetime.now().strftime('%Y%m%d')}.xlsx"
    df.to_excel(out, index=False)
    present = df[df["Status"] == "Present"].shape[0]
    print(f"\nDone! {present}/{len(df)} present. File saved: {out}")

if __name__ == "__main__":
    run()