import cv2
from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)  # Allow all origins during development

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def dehaze(frame, omega=0.95, tmin=0.1, gamma=1.0, color_balance=None):
    # Step 1: Calculate the dark channel
    min_channel = np.min(frame, axis=2)
    dark_channel = cv2.erode(min_channel, np.ones((15, 15), np.uint8))
    
    # Step 2: Estimate the atmospheric light
    num_pixels = dark_channel.size
    num_top_pixels = int(num_pixels * omega)
    flat_dark_channel = dark_channel.flatten()
    indices = np.argpartition(flat_dark_channel, -num_top_pixels)[-num_top_pixels:]
    atmospheric_light = np.mean(frame.reshape(-1, 3)[indices], axis=0) * 0.8  # Use a fraction of the mean
    
    # Step 3: Calculate the transmission map
    transmission = 1 - omega * min_channel / atmospheric_light.max()
    transmission[transmission < tmin] = tmin
    
    # Step 4: Recover the haze-free image
    recovered_image = np.zeros_like(frame, dtype=np.float32)
    for i in range(3):
        recovered_image[:, :, i] = ((frame[:, :, i].astype(np.float32) - atmospheric_light[i]) /
                                    transmission + atmospheric_light[i])
    
    # Gamma correction
    recovered_image = np.clip(recovered_image, 0, 255).astype(np.uint8)
    recovered_image = np.power(recovered_image / 255.0, 1 / gamma) * 255.0
    
    # Color balance
    if color_balance is not None:
        recovered_image = cv2.xphoto.createSimpleWB().balanceWhite(recovered_image.astype(np.uint8))  # Convert to compatible data type
    
    return recovered_image

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        dehazed_frame = dehaze(frame, omega=0.5, tmin=0.1, gamma=1.5, color_balance=True)

        ret, buffer = cv2.imencode('.jpg', dehazed_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def dehaze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    print("Processing")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        dehazed_frame = dehaze(frame, omega=0.5, tmin=0.1, gamma=1.5, color_balance=True)
        frames.append(dehazed_frame)

    cap.release()
    print("Processed")
    # Combine frames into a video
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in frames:
        out.write(frame)
    
    out.release()

    # Create a window to display the processed video
    window_title = 'Processed Video'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, width // 2, height // 2)  # Resize the window to 1/4 of the screen
    cv2.moveWindow(window_title, width // 4, height // 4)  # Position the window in the center
    cv2.imshow(window_title, frames[0])

    # Add a close button to the window
    def close_window(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.destroyWindow(window_title)

    cv2.setMouseCallback(window_title, close_window)

    cv2.waitKey(0)  # Wait until a key is pressed or the window is closed

    return 'output.mp4'


@app.route('/')
def index():
    return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    image_data = data['image_data'].split(',')[1]  # Remove the "data:image/jpeg;base64," prefix
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform dehazing on the frame
    dehazed_frame = dehaze(frame, omega=0.5, tmin=0.05, gamma=1.5, color_balance=True)

    # Encode the dehazed frame to base64 for sending to the client
    _, buffer = cv2.imencode('.jpg', dehazed_frame)
    dehazed_image_data = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'dehazed_image': dehazed_image_data})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video_file' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video_file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Save and process the uploaded video file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Perform dehazing on the uploaded video and display in a separate window
        dehaze_video(filename)

        # Return a success message
        return jsonify({'message': 'Video processed successfully'})

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
