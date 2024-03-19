

def dehaze(frame, omega=0.95, tmin=0.1, gamma=1.0, color_balance=None):
    # Step 1: Calculate the dark channel
    min_channel = np.min(frame, axis=2)
    dark_channel = cv2.erode(min_channel, np.ones((15, 15), np.uint8))
    
    # Step 2: Adjust atmospheric light estimation for fire
    is_fire_scene = True  # Placeholder for a condition to detect fire scenes
    if is_fire_scene:
        atmospheric_light = np.array([frame[:,:,2].max(), frame[:,:,1].max(), frame[:,:,0].max()]) * 0.9
    else:
        num_pixels = dark_channel.size
        num_top_pixels = int(num_pixels * omega)
        flat_dark_channel = dark_channel.flatten()
        indices = np.argpartition(flat_dark_channel, -num_top_pixels)[-num_top_pixels:]
        atmospheric_light = np.mean(frame.reshape(-1, 3)[indices], axis=0) * 0.8
    
    # Step 3: Calculate the transmission map
    transmission = 1 - omega * min_channel / atmospheric_light.max()
    transmission[transmission < tmin] = tmin
    
    # Step 4: Recover the haze-free image
    recovered_image = np.zeros_like(frame, dtype=np.float32)
    for i in range(3):
        recovered_image[:, :, i] = ((frame[:, :, i].astype(np.float32) - atmospheric_light[i]) /
                                    transmission + atmospheric_light[i])
    
    # Apply gamma correction specifically tuned for fire conditions
    recovered_image = np.clip(recovered_image, 0, 255).astype(np.uint8)
    recovered_image = apply_gamma_correction(recovered_image, gamma=gamma)
    
     # Color balance
    if color_balance is not None:
        recovered_image = cv2.xphoto.createSimpleWB().balanceWhite(recovered_image.astype(np.uint8))  # Convert to compatible data type
    
    
    # Post-processing: Enhance contrast and color for better visibility in fire scenes
    # TODO: Implement enhancements
    
    return recovered_image

def apply_gamma_correction(image, gamma=0.8):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)  # Allow all origins during development

def dehaze(frame, omega=0.95, tmin=0.1, gamma=1.0, color_balance=None):
    # Step 1: Calculate the dark channel
    min_channel = np.min(frame, axis=2)
    dark_channel = cv2.erode(min_channel, np.ones((15, 15), np.uint8))
    
    # Step 2: Adjust atmospheric light estimation for fire
    is_fire_scene = True  # Placeholder for a condition to detect fire scenes
    if is_fire_scene:
        atmospheric_light = np.array([frame[:,:,2].max(), frame[:,:,1].max(), frame[:,:,0].max()]) * 0.9
    else:
        num_pixels = dark_channel.size
        num_top_pixels = int(num_pixels * omega)
        flat_dark_channel = dark_channel.flatten()
        indices = np.argpartition(flat_dark_channel, -num_top_pixels)[-num_top_pixels:]
        atmospheric_light = np.mean(frame.reshape(-1, 3)[indices], axis=0) * 0.8
    
    # Step 3: Calculate the transmission map
    transmission = 1 - omega * min_channel / atmospheric_light.max()
    transmission[transmission < tmin] = tmin
    
    # Step 4: Recover the haze-free image
    recovered_image = np.zeros_like(frame, dtype=np.float32)
    for i in range(3):
        recovered_image[:, :, i] = ((frame[:, :, i].astype(np.float32) - atmospheric_light[i]) /
                                    transmission + atmospheric_light[i])
    
    # Apply gamma correction specifically tuned for fire conditions
    recovered_image = np.clip(recovered_image, 0, 255).astype(np.uint8)
    recovered_image = apply_gamma_correction(recovered_image, gamma=gamma)
    
     # Color balance
    if color_balance is not None:
        recovered_image = cv2.xphoto.createSimpleWB().balanceWhite(recovered_image.astype(np.uint8))  # Convert to compatible data type
    
    
    # Post-processing: Enhance contrast and color for better visibility in fire scenes
    # TODO: Implement enhancements
    
    return recovered_image

def apply_gamma_correction(image, gamma=0.8):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

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

@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True,port=5001)

