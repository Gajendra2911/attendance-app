import os
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"

import cv2
import numpy as np
import pandas as pd
import cloudinary
import cloudinary.uploader
import cloudinary.api
from flask import Flask, request, jsonify, render_template_string
from deepface import DeepFace
from datetime import datetime
import tempfile, requests

app = Flask(__name__)

cloudinary.config(
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", "your_cloud_name"),
    api_key    = os.environ.get("CLOUDINARY_API_KEY",    "your_api_key"),
    api_secret = os.environ.get("CLOUDINARY_API_SECRET", "your_api_secret")
)

RECORDS_PATH = "attendance_records"
os.makedirs(RECORDS_PATH, exist_ok=True)

HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Attendance System</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto; padding: 16px; background: #f5f5f5; }
    h2 { color: #2c3e50; text-align: center; margin-bottom: 4px; }
    .subtitle { text-align:center; color:#888; font-size:13px; margin-bottom:16px; }
    .tabs { display:flex; gap:8px; margin-bottom:16px; }
    .tab { flex:1; padding:10px; text-align:center; border-radius:8px; border:1px solid #ddd; background:#fff; cursor:pointer; font-size:14px; font-weight:500; color:#666; }
    .tab.active { background:#2ecc71; color:#fff; border-color:#2ecc71; }
    .section { display:none; }
    .section.active { display:block; }
    input, select { width:100%; padding:10px; margin:6px 0 12px; font-size:15px; border-radius:8px; border:1px solid #ccc; }
    button { width:100%; padding:12px; font-size:15px; border-radius:8px; border:none; cursor:pointer; font-weight:bold; margin-top:4px; }
    .btn-green { background:#2ecc71; color:#fff; }
    .btn-blue  { background:#3498db; color:#fff; }
    .photo-row { display:flex; gap:8px; margin:8px 0; }
    .photo-slot { flex:1; aspect-ratio:1; border-radius:8px; border:2px dashed #ccc; display:flex; align-items:center; justify-content:center; flex-direction:column; font-size:11px; color:#aaa; cursor:pointer; background:#fafafa; }
    .photo-slot.done { border-color:#2ecc71; background:#eafaf1; color:#2ecc71; }
    .photo-slot img { width:100%; height:100%; object-fit:cover; border-radius:6px; }
    #result, #enroll-result { background:#fff; border-radius:8px; padding:14px; margin-top:14px; display:none; }
    .row { display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f0f0f0; font-size:13px; }
    .present { color:#27ae60; font-weight:bold; }
    .absent  { color:#e74c3c; font-weight:bold; }
    #status, #enroll-status { text-align:center; color:#888; font-size:13px; margin-top:8px; min-height:20px; }
    .hint { font-size:12px; color:#aaa; text-align:center; margin:4px 0 10px; }
  </style>
</head>
<body>
  <h2>Attendance System</h2>
  <p class="subtitle">ICA Skill Development</p>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('attendance')">Mark attendance</div>
    <div class="tab" onclick="switchTab('enroll')">Enroll student</div>
  </div>

  <div class="section active" id="section-attendance">
    <input type="text" id="batch" placeholder="Batch ID (e.g. ICA-B01)">
    <input type="text" id="session" value="Morning" placeholder="Session">
    <input type="file" id="photo" accept="image/*" capture="environment">
    <p class="hint">Take a group photo of the whole class</p>
    <button class="btn-green" onclick="markAttendance()">Mark attendance</button>
    <div id="status"></div>
    <div id="result"></div>
  </div>

  <div class="section" id="section-enroll">
    <input type="text" id="e-roll" placeholder="Roll number (e.g. ICA003)">
    <input type="text" id="e-name" placeholder="Student full name">
    <input type="text" id="e-batch" placeholder="Batch ID (e.g. ICA-B01)">
    <p class="hint">Take 3 clear photos — front, slight left, slight right</p>
    <div class="photo-row">
      <div class="photo-slot" id="slot0" onclick="takePhoto(0)">
        <span style="font-size:24px">+</span><span>Photo 1</span>
      </div>
      <div class="photo-slot" id="slot1" onclick="takePhoto(1)">
        <span style="font-size:24px">+</span><span>Photo 2</span>
      </div>
      <div class="photo-slot" id="slot2" onclick="takePhoto(2)">
        <span style="font-size:24px">+</span><span>Photo 3</span>
      </div>
    </div>
    <input type="file" id="photo-input" accept="image/*" capture="user" style="display:none" onchange="photoSelected(this)">
    <button class="btn-blue" onclick="enrollStudent()">Save student</button>
    <div id="enroll-status"></div>
    <div id="enroll-result"></div>
  </div>

<script>
const photos = [null, null, null];
let currentSlot = 0;

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', ['attendance','enroll'][i]===tab));
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.getElementById('section-'+tab).classList.add('active');
}

function takePhoto(slot) {
  currentSlot = slot;
  document.getElementById('photo-input').click();
}

function photoSelected(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  photos[currentSlot] = file;
  const slot = document.getElementById('slot'+currentSlot);
  const reader = new FileReader();
  reader.onload = e => {
    slot.innerHTML = '<img src="'+e.target.result+'">';
    slot.classList.add('done');
  };
  reader.readAsDataURL(file);
  input.value = '';
}

async function enrollStudent() {
  const roll  = document.getElementById('e-roll').value.trim();
  const name  = document.getElementById('e-name').value.trim();
  const batch = document.getElementById('e-batch').value.trim();
  if (!roll || !name || !batch) { alert('Please fill in all fields'); return; }
  if (!photos[0] || !photos[1] || !photos[2]) { alert('Please take all 3 photos'); return; }
  document.getElementById('enroll-status').textContent = 'Saving student... please wait';
  const fd = new FormData();
  fd.append('roll', roll); fd.append('name', name); fd.append('batch', batch);
  photos.forEach((p,i) => fd.append('photo'+i, p));
  try {
    const res  = await fetch('/enroll', { method:'POST', body:fd });
    const data = await res.json();
    document.getElementById('enroll-status').textContent = '';
    const box = document.getElementById('enroll-result');
    if (data.error) { box.innerHTML='<p style="color:red">Error: '+data.error+'</p>'; }
    else { box.innerHTML='<p style="color:#27ae60;font-weight:bold;text-align:center">'+name+' enrolled successfully!</p>'; }
    box.style.display='block';
  } catch(e) { document.getElementById('enroll-status').textContent='Something went wrong. Try again.'; }
}

async function markAttendance() {
  const batch   = document.getElementById('batch').value.trim();
  const session = document.getElementById('session').value.trim();
  const file    = document.getElementById('photo').files[0];
  if (!batch || !file) { alert('Please enter Batch ID and take a photo'); return; }
  document.getElementById('status').textContent = 'Recognising faces... please wait (30-60 sec)';
  const fd = new FormData();
  fd.append('photo', file); fd.append('batch', batch); fd.append('session', session);
  try {
    const res  = await fetch('/mark', { method:'POST', body:fd });
    const data = await res.json();
    document.getElementById('status').textContent = '';
    if (data.error) { alert('Error: '+data.error); return; }
    let html = '<h3 style="margin:0 0 10px">Results — '+data.date+'</h3>'
      +'<div class="row"><b>Batch</b><span>'+data.batch+'</span></div>'
      +'<div class="row"><b>Present</b><span class="present">'+data.present+'</span></div>'
      +'<div class="row"><b>Absent</b><span class="absent">'+data.absent+'</span></div>'
      +'<div class="row"><b>Attendance</b><span>'+data.percentage+'%</span></div><hr style="margin:10px 0">';
    data.records.forEach(r => {
      html += '<div class="row"><span>'+r.roll+' — '+r.name+'</span>'
             +'<span class="'+(r.status==='Present'?'present':'absent')+'">'+r.status+'</span></div>';
    });
    const box = document.getElementById('result');
    box.innerHTML = html; box.style.display = 'block';
  } catch(e) { document.getElementById('status').textContent='Something went wrong. Try again.'; }
}
</script>
</body>
</html>
"""

def download_enrolled_db():
    """Download enrolled photos from Cloudinary to local temp folder."""
    db_path = "/tmp/enrolled_db"
    os.makedirs(db_path, exist_ok=True)
    try:
        result = cloudinary.api.resources(type="upload", prefix="enrolled_db/", max_results=500)
        for resource in result.get("resources", []):
            public_id = resource["public_id"]
            parts = public_id.replace("enrolled_db/", "").split("/")
            if len(parts) == 2:
                folder_name, filename = parts
                folder_path = os.path.join(db_path, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                file_path = os.path.join(folder_path, filename + ".jpg")
                if not os.path.exists(file_path):
                    img_data = requests.get(resource["url"]).content
                    with open(file_path, "wb") as f:
                        f.write(img_data)
    except Exception as e:
        print(f"Error downloading enrolled DB: {e}")
    return db_path

@app.route("/")
def home():
    return render_template_string(HOME_PAGE)

@app.route("/enroll", methods=["POST"])
def enroll():
    try:
        roll  = request.form.get("roll", "").strip()
        name  = request.form.get("name", "").strip().replace(" ", "_")
        batch = request.form.get("batch", "").strip()
        if not roll or not name:
            return jsonify({"error": "Roll number and name are required"})
        folder_name = f"{roll}_{name}"
        for i in range(3):
            file = request.files.get(f"photo{i}")
            if file:
                cloudinary.uploader.upload(
                    file,
                    public_id=f"enrolled_db/{folder_name}/photo{i+1}",
                    overwrite=True
                )
        return jsonify({"success": True, "message": f"{name} enrolled successfully"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/mark", methods=["POST"])
def mark():
    try:
        batch   = request.form.get("batch", "BATCH")
        session = request.form.get("session", "Morning")
        file    = request.files["photo"]
        img_array = np.frombuffer(file.read(), np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        tmp_photo = f"/tmp/group_{datetime.now().strftime('%Y%m%d_%H%M')}.jpg"
        cv2.imwrite(tmp_photo, frame)
        cloudinary.uploader.upload(tmp_photo,
            public_id=f"group_photos/{batch}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            overwrite=True)

        db_path    = download_enrolled_db()
        candidates = []
        for folder in os.listdir(db_path):
            parts = folder.split("_", 1)
            if len(parts) == 2:
                candidates.append({"roll": parts[0], "name": parts[1].replace("_", " ")})

        matched = set()
        faces = DeepFace.extract_faces(tmp_photo, enforce_detection=False)
        for i, face in enumerate(faces):
            face_img = (face["face"] * 255).astype("uint8")
            tmp = f"/tmp/_face_{i}.jpg"
            cv2.imwrite(tmp, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            try:
                result = DeepFace.find(tmp, db_path=db_path,
                                       model_name="Facenet512",
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
        out = f"/tmp/{batch}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        df.to_excel(out, index=False)
        cloudinary.uploader.upload(out,
            public_id=f"attendance_records/{batch}_{datetime.now().strftime('%Y%m%d')}",
            resource_type="raw", overwrite=True)

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