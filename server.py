from flask import Flask, request, jsonify, render_template_string
from deepface import DeepFace
from datetime import datetime
import pandas as pd
import os, cv2, base64, numpy as np

app = Flask(__name__)
DB_PATH = "enrolled_db"
RECORDS_PATH = "attendance_records"
os.makedirs(RECORDS_PATH, exist_ok=True)

HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attendance App</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
    h2 { color: #2c3e50; text-align: center; }
    input, select, button { width: 100%; padding: 12px; margin: 8px 0; font-size: 16px; border-radius: 8px; border: 1px solid #ccc; box-sizing: border-box; }
    button { background: #2ecc71; color: white; border: none; cursor: pointer; font-weight: bold; }
    button:active { background: #27ae60; }
    #preview { width: 100%; border-radius: 8px; margin: 10px 0; display: none; }
    #result { background: white; border-radius: 8px; padding: 16px; margin-top: 16px; display: none; }
    .present { color: #27ae60; font-weight: bold; }
    .absent  { color: #e74c3c; font-weight: bold; }
    .row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #eee; font-size: 14px; }
  </style>
</head>
<body>
  <h2>Attendance System</h2>
  <input type="text" id="batch" placeholder="Enter Batch ID (e.g. ICA-B01)">
  <input type="text" id="session" placeholder="Session (e.g. Morning)" value="Morning">
  <input type="file" id="photo" accept="image/*" capture="environment" onchange="preview(this)">
  <img id="preview" src="">
  <button onclick="submit()">Mark Attendance</button>
  <div id="status" style="text-align:center;color:#888;margin-top:8px"></div>
  <div id="result"></div>

<script>
function preview(input) {
  if (input.files && input.files[0]) {
    const reader = new FileReader();
    reader.onload = e => {
      document.getElementById('preview').src = e.target.result;
      document.getElementById('preview').style.display = 'block';
    };
    reader.readAsDataURL(input.files[0]);
  }
}

async function submit() {
  const batch   = document.getElementById('batch').value.trim();
  const session = document.getElementById('session').value.trim();
  const file    = document.getElementById('photo').files[0];
  if (!batch || !file) { alert('Please enter Batch ID and select a photo'); return; }

  document.getElementById('status').textContent = 'Recognising faces... please wait (30-60 sec)';
  const formData = new FormData();
  formData.append('photo', file);
  formData.append('batch', batch);
  formData.append('session', session);

  try {
    const res  = await fetch('/mark', { method: 'POST', body: formData });
    const data = await res.json();
    document.getElementById('status').textContent = '';
    if (data.error) { alert('Error: ' + data.error); return; }

    let html = `<h3 style="margin:0 0 10px">Results — ${data.date}</h3>
      <div class="row"><b>Batch</b><span>${data.batch}</span></div>
      <div class="row"><b>Present</b><span class="present">${data.present}</span></div>
      <div class="row"><b>Absent</b><span class="absent">${data.absent}</span></div>
      <div class="row"><b>Attendance</b><span>${data.percentage}%</span></div>
      <hr style="margin:12px 0">`;
    data.records.forEach(r => {
      html += `<div class="row"><span>${r.roll} — ${r.name}</span>
               <span class="${r.status === 'Present' ? 'present' : 'absent'}">${r.status}</span></div>`;
    });
    const result = document.getElementById('result');
    result.innerHTML = html;
    result.style.display = 'block';
  } catch(e) {
    document.getElementById('status').textContent = 'Something went wrong. Try again.';
  }
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HOME_PAGE)

@app.route("/mark", methods=["POST"])
def mark():
    try:
        batch   = request.form.get("batch", "BATCH")
        session = request.form.get("session", "Morning")
        file    = request.files["photo"]

        img_array = np.frombuffer(file.read(), np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        photo_path = f"group_photos/{batch}_{datetime.now().strftime('%Y%m%d_%H%M')}.jpg"
        os.makedirs("group_photos", exist_ok=True)
        cv2.imwrite(photo_path, frame)

        candidates = []
        for folder in os.listdir(DB_PATH):
            parts = folder.split("_", 1)
            if len(parts) == 2:
                candidates.append({"roll": parts[0], "name": parts[1].replace("_", " ")})

        matched = set()
        faces = DeepFace.extract_faces(photo_path, enforce_detection=False)
        for i, face in enumerate(faces):
            face_img = (face["face"] * 255).astype("uint8")
            tmp = f"_tmp_{i}.jpg"
            cv2.imwrite(tmp, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            try:
                result = DeepFace.find(tmp, db_path=DB_PATH, model_name="Facenet512",
                                       enforce_detection=False, silent=True)
                if result and len(result[0]) > 0:
                    if result[0].iloc[0]["distance"] < 0.4:
                        folder = os.path.normpath(result[0].iloc[0]["identity"]).split(os.sep)[-2]
                        matched.add(folder.split("_")[0])
            except: pass
            if os.path.exists(tmp): os.remove(tmp)

        today   = datetime.now().strftime("%d-%m-%Y")
        records = [{"roll": c["roll"], "name": c["name"],
                    "status": "Present" if c["roll"] in matched else "Absent"}
                   for c in candidates]

        df = pd.DataFrame([{"Roll No": r["roll"], "Name": r["name"],
                             "Status": r["status"], "Date": today,
                             "Batch": batch, "Session": session}
                            for r in records])
        out = f"{RECORDS_PATH}/{batch}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        df.to_excel(out, index=False)

        present = sum(1 for r in records if r["status"] == "Present")
        return jsonify({
            "batch": batch, "date": today,
            "present": present, "absent": len(records) - present,
            "percentage": round(present / len(records) * 100) if records else 0,
            "records": records
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)