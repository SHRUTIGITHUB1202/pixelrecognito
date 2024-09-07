from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
import os
import face_recognition
import cv2
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
import json
app = Flask(__name__)

camera = cv2.VideoCapture(0)
app.secret_key = 'your_secret_key'  # Change this to a random secret key
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# List to store known face encodings and names
known_face_encodings = []
known_face_names = []

# List to store attendance records
past_attendance = {}

@app.route('/')
def index():
    return render_template('index.html')

from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
import os
import face_recognition
import cv2
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
import pickle

app = Flask(__name__)

camera = cv2.VideoCapture(0)
app.secret_key = 'your_secret_key'  # Change this to a random secret key
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# List to store known face encodings and names
known_face_encodings = []
known_face_names = []
REGISTERED_FACES_FILE = 'registered_faces.pkl'
ATTENDANCE_FILE = 'attendance_records.pkl'
# List to store attendance records
past_attendance = {}
if os.path.exists(REGISTERED_FACES_FILE):
    with open(REGISTERED_FACES_FILE, 'rb') as file:
        registered_faces_data = pickle.load(file)
        known_face_encodings, known_face_names = registered_faces_data.get('encodings', []), registered_faces_data.get('names', [])


# Load attendance records from file if available
if os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'rb') as file:
        attendance_records = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')

        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            # Ensure that the upload folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # Save the image to the upload folder
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(image_path)

            # Load the image for face recognition
            new_face_image = face_recognition.load_image_file(image_path)

            # Get the face encoding for the new face
            new_face_encoding = face_recognition.face_encodings(new_face_image)

            if len(new_face_encoding) > 0:
                # Add the new face encoding and name to known_face_encodings and known_face_names
                known_face_encodings.append(new_face_encoding[0])
                known_face_names.append(name)

                # Save registered faces to file
                save_registered_faces()

                # Print debugging information
                print("Number of known faces:", len(known_face_encodings))
                print("Known face names:", known_face_names)

                flash('Registration successful!', 'success')
                return redirect(url_for('login'))

            else:
                flash('No face found in the provided image. Please try again with a different image.', 'danger')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = "Unknown"

        while True:
            success, frame = camera.read()
            if not success:
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            if name != "Unknown":
                session['user_name'] = name
                mark_attendance(name)
                flash(f'Attendance marked for {name}', 'success')
                return redirect(url_for('mark_attendance_page'))

    return render_template('login.html')


# ... (rest of your code)


def generate_frames(camera, known_face_encodings, known_face_names):
    while True:
        success, frame = camera.read()
        if not success:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if any(matches):
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def save_registered_faces():
    data_to_save = {'encodings': known_face_encodings, 'names': known_face_names}
    with open(REGISTERED_FACES_FILE, 'wb') as file:
        pickle.dump(data_to_save, file)

def save_attendance_records():
    with open(ATTENDANCE_FILE, 'wb') as file:
        pickle.dump(attendance_records, file)

# ... (previous code)

def mark_attendance(name):
    if 'user_name' in session:
        user_name = session['user_name']
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

        # Add the attendance record to the dictionary
        if user_name not in attendance_records:
            attendance_records[user_name] = []
        attendance_records[user_name].append({'date': now.strftime("%Y-%m-%d"), 'time': now.strftime("%H:%M:%S")})

        # Save updated attendance records to file
        save_attendance_records()

        flash(f'Attendance marked for {user_name} at {dt_string}', 'success')
        return render_template('mark_attendance.html', user_name=user_name, time=dt_string,
                               past_records=attendance_records[user_name])
    else:
        return redirect(url_for('login'))

@app.route('/mark_attendance_page')
def mark_attendance_page():
    if 'user_name' in session:
        user_name = session['user_name']
        return render_template('mark_attendance.html', user_name=user_name,
                               past_records=attendance_records.get(user_name, []))
    else:
        return redirect(url_for('login'))



# List to store attendance records
attendance_records = {}

def save_attendance_records():
    # Save attendance records to file (you can customize this part based on your needs)
    with open('attendance_records.json', 'w') as file:
        json.dump(attendance_records, file)

def load_attendance_records():
    global attendance_records
    try:
        # Load attendance records from file
        with open('attendance_records.json', 'r') as file:
            attendance_records = json.load(file)
    except FileNotFoundError:
        attendance_records = {}

# Load attendance records at the start of the application
load_attendance_records()
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(camera, known_face_encodings, known_face_names),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
