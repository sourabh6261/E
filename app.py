import cv2.data
from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for, send_file
import cv2
import face_recognition
import json
import csv
import os
import datetime
import uuid  # For generating unique filenames
from collections import defaultdict
from datetime import time
import io

# Initialize the text-to-speech engine


# Function to speak the given message

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Paths for user data and attendance file
user_data_path = "face.json"
attendance_file_path = "Attendance.csv"
user_images_folder = "user_images"  # Folder to store user images inside static

# Initialize user data dictionary for user data
user_data = {}

# Create user images folder if it doesn't exist
if not os.path.exists(user_images_folder):
    os.makedirs(user_images_folder)

# Initialize user data dictionary for user data
user_data = {}
# Create Attendance.csv file if it doesn't exist and write header row
if os.path.exists(user_data_path):
    with open(user_data_path, "r") as file:
        user_data = json.load(file)

if not os.path.exists(attendance_file_path):
    with open(attendance_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Emp Name","Department", "EmpId", "Entering Time", "Leaving Time", "Updated Entry Time", "Updated Leaving Time"])

# Initialize video capture from webcam
video_capture = cv2.VideoCapture(1)
# Set the refresh rate

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Function to generate video frames from webcam feed
def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
            for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# Route for capturing and registering a new user
@app.route("/register", methods=["GET", "POST"])
def register_user():
    if request.method == "POST":
        # Handle form submission here
        data = request.form
        name = data.get("name")
        contact = data.get("contact")
        department = data.get("department")
        employ_id = data.get("employ_id")
        # Process the user's details
        
        # Capture a frame from the video feed
        ret, frame = video_capture.read()
        if not ret:
            return jsonify({"message": "Error capturing photo from webcam"}), 500
        
        # Convert frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize face detector
        face_locations = face_recognition.face_locations(frame_rgb)
        # If no face detected, return error message
        if not face_locations:
            return jsonify({"message": "No face detected in the photo"}), 400
        # If multiple faces detected, return error message
        elif len(face_locations) > 1:
            return jsonify({"message": "Multiple faces detected. Please ensure only one face is in the frame."}), 400
        for (top,right,left,bottom) in face_locations:
            cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
        
        
        # Save the captured image with a unique filename inside the static folder
        filename = os.path.join("user_images", str(uuid.uuid4()) + ".jpg")
        image_path = os.path.join(app.root_path,"static", filename)
        cv2.imwrite(image_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        face_encoding = face_recognition.face_encodings(frame_rgb,face_locations)[0]
        
        # If face embedding is not detected, return error message
        if face_encoding is None:
            return jsonify({"message": "Face could not be detected in the provided image."}), 400
        
        # Check for similar faces in existing user data
        for existing_user, user_info in user_data.items():
            existing_encoding = user_info.get("encoding")
            if existing_encoding is not None:
                distance = face_recognition.face_distance([existing_encoding],face_encoding)[0]
                if distance < 0.4:
                    return jsonify({"message": f"Similar face already registered as {existing_user}."})
        
        # Add new user data to user_data dictionary
        user_data[name] = {
            "contact": contact,
            "employ_id": employ_id,
            "department": department,
            "image": filename,  # Store filename of the image
            "encoding": face_encoding.tolist() # Convert embedding to list
            
        }
        
        # Write updated user data to file
        with open(user_data_path, "w") as file:
            json.dump(user_data, file, indent=4)
        
        # Redirect to registration page with success parameter
        return redirect(url_for("registration_success", message=f"User {name} registered successfully"))
    else:
        # Render the registration page
        return render_template("register_user.html")

# Route for displaying registration success popup
@app.route("/registration_success")
def registration_success():
    message = request.args.get("message")
    return f"""
    <script>
        alert("{message}");
        window.location.href = "/";
    </script>
    """

# Initialize a dictionary to store recognition times for each recognized user
recognized_times = defaultdict(list)

# Route for recognizing a user
@app.route("/recognize")
def recognize_user():
    # Capture a frame from the video feed
    ret, frame = video_capture.read()
    if not ret:
        return jsonify({"message": "Error capturing photo from webcam"}), 500
    
    # Convert frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(frame_rgb)

    # If no face detected, return error message
    if not face_locations:
        return jsonify({"message": "No face detected in the photo"}), 400
    
    face_encoding = face_recognition.face_encodings(frame_rgb, face_locations)[0]
    
    # Initialize variables for recognized user and minimum distance
    recognized_user = None
    min_distance = 0.4
    
    # Iterate over each face detected in the frame
    for (top, right, bottom, left) in face_locations:
        face_image = frame_rgb[top:bottom, left:right]
 
        # Compare face embedding with registered users' embeddings
        for name, user_info in user_data.items():
            employ_id = user_info["employ_id"]
            registered_encoding = user_info["encoding"]
            distance = face_recognition.face_distance([registered_encoding], face_encoding)[0]
            
            # Update recognized user if distance is smaller than minimum distance
            if distance < min_distance:
                min_distance = distance
                recognized_user = name
    
    # If recognized user found
    if recognized_user:
        # Check if the user has been recognized multiple times within a minute
        current_time = datetime.datetime.now()
        if recognized_times[recognized_user]:
            last_recognition_time = recognized_times[recognized_user][-1]
            time_difference = (current_time - last_recognition_time).total_seconds()
            if time_difference < 60:
                # User recognized within a minute, do not mark attendance again
                return jsonify({"message": f"Attendance already taken for {recognized_user} within the last minute."}), 200
        
        # Update recognized times for the user
        recognized_times[recognized_user].append(current_time)
        
        current_date = current_time.strftime("%Y-%m-%d")
        current_time_str = current_time.strftime("%H:%M:%S")

        # Read existing attendance data
        with open(attendance_file_path, "r") as file:
            reader = csv.DictReader(file)
            columns = reader.fieldnames
            rows = list(reader)
        
        # Check if user already has an entry for the current date
        found_user = False
        for row in rows:
            if row["Date"] == current_date and row["EmpId"] == user_data[recognized_user]["employ_id"]:
                found_user = True
                if not row.get("Entering Time"):
                    row["Entering Time"] = current_time_str
                elif not row.get("Leaving Time"):
                    row["Leaving Time"] = current_time_str
                elif not row.get("Updated Entry Time"):
                    row["Updated Entry Time"] = current_time_str
                else:
                    row["Updated Leaving Time"] = current_time_str
                break
        
        # If user not found for the current date, add new entry
        if not found_user:
            new_row = {
                "Date": current_date,
                "Emp Name": recognized_user,
                "EmpId": user_data[recognized_user]["employ_id"],
                "Department": user_data[recognized_user]["department"],
                "Entering Time": current_time_str}
            rows.append(new_row)

        with open(attendance_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        return jsonify({"message": f"User {recognized_user} recognized. Attendance marked.", "image": user_data[recognized_user]["image"]}), 200
    else:
        return jsonify({"message": "User not recognized"}), 404
    
# Route for rendering admin login page
@app.route("/admin_login", methods=["POST", "GET"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "admin123":
            session['logged_in'] = True
            return redirect(url_for("admin_dashboard"))
        else:
            return jsonify({"message": "Invalid credentials"}), 401
    else:
        return render_template("admin_login.html")

# Route for rendering admin dashboard
@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('admin_login'))

    attendance_data = []
    with open(attendance_file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            attendance_data.append(row)

    return render_template("admin_dashboard.html", attendance_data=attendance_data)

# Route for searching employee in attendance records
@app.route("/search")
def search_employee():
    name = request.args.get("name", "").lower()
    from_date = request.args.get("fromDate")
    to_date = request.args.get("toDate")
    all_data = request.args.get("all_data")  # Check if user wants all data
    
    attendance_data = []
    with open(attendance_file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            attendance_data.append(row)
    
    if all_data:
        results = [row for row in attendance_data if name in row["Emp Name"].lower()]
    elif from_date and to_date:
        results = [row for row in attendance_data if name in row["Emp Name"].lower() and from_date <= row["Date"] <= to_date]
    else:
        results = []

    return jsonify({"results": results})


@app.route("/download", methods=["GET"])
def download_attendance():
    from_date = request.args.get("fromDate")
    to_date = request.args.get("toDate")
    download_type = request.args.get("download_type")
    department = request.args.get("department")
    employee_name = request.args.get("employee_name")
    
    with open(attendance_file_path, "r") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    
    if download_type == "department-wise":
        filtered_rows = [row for row in rows if row["Department"] == department]
    elif download_type == "employee-wise":
        filtered_rows = [row for row in rows if row["Emp Name"] == employee_name]
    else:
        filtered_rows = rows
    
    if from_date and to_date:
        filtered_rows = [row for row in filtered_rows if from_date <= row["Date"] <= to_date]
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(filtered_rows)
    
    output.seek(0)
    
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=Filtered_Attendance.csv"}
    )

# Route for deleting user from user data and attendance records
@app.route("/delete_user", methods=["POST"])
def delete_user():
    emp_id = request.form.get("emp_id")
    if emp_id:
        # Remove user from user_data
        user_to_delete = None
        for name, user_info in user_data.items():
            if user_info["employ_id"] == emp_id:
                user_to_delete = name
                break
        if user_to_delete:
            del user_data[user_to_delete]
            # Write updated user data to file
            with open(user_data_path, "w") as file:
                json.dump(user_data, file, indent=4)
        
        # Remove user's attendance records
        with open(attendance_file_path, "r") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
        updated_rows = [row for row in rows if row["EmpId"] != emp_id]
        with open(attendance_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        
        return redirect(url_for("admin_dashboard"))
    else:
        return jsonify({"message": "Employee ID not provided"}), 400

@app.route("/download_employee_details", methods=["GET"])
def download_employee_details():
    # Retrieve employee details from user data
    employee_details = []
    for name, info in user_data.items():
        employee_detail = {
            "Emp Name": name,
            "Department": info["department"],
            "EmpId": info["employ_id"],
            # You can add more fields here if needed
        }
        employee_details.append(employee_detail)
    
    # Prepare CSV file
    output = io.StringIO()
    fieldnames = ["Emp Name", "Department", "EmpId"]  # Define fieldnames for the CSV file
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(employee_details)
    
    output.seek(0)
    
    # Return the CSV file as a response
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=employee_details.csv"}
    )

# Route for filtering attendance records
@app.route("/filter")
def filter_attendance():
    filter_date = request.args.get("date")
    filter_department = request.args.get("department")
    attendance_data = []

    # Read attendance data from CSV file
    with open(attendance_file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            attendance_data.append(row)

    # Filter attendance data based on provided date and department
    filtered_results = [row for row in attendance_data if row["Date"] == filter_date and row["Department"] == filter_department]

    return jsonify(filtered_results)


def get_department_status(department):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    total_employees = 0
    present_employees = 0

    # Count total employees in the department
    with open("employee_details.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["Department"] == department:
                total_employees += 1

    # Count present employees for the current date
    with open("Attendance.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["Date"] == current_date and row["Department"] == department:
                present_employees += 1

    status = f"{present_employees}/{total_employees}"
    
    return status


def generate_department_status(department):
    while True:
        department_status = {department: get_department_status(department)}
        yield "data: {}\n\n".format(json.dumps(department_status))
        time.sleep(1)  # Adjust the delay as needed

# Route for department status updates
@app.route("/department_status/<department>")
def department_status(department):
    status = get_department_status(department)
    return jsonify({"status": status})



@app.route('/user_side')
def user_side():
    return render_template('user_side_screen.html')


# Route for rendering homepage
@app.route("/")
def index():
    return render_template("homepage.html")

if __name__ == "__main__":
    app.run(debug=True)











