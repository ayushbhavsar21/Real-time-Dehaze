# Import necessary libraries
import cv2
import numpy as np
import math

# Dehaze function goes here
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
# Main video processing loop
if __name__ == '__main__':
    cap = cv2.VideoCapture('../assets/fire.mp4')

    screen_width = 1920
    screen_height = 1080
    resize_width = screen_width // 4
    resize_height = screen_height // 4

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (resize_width, resize_height))
        
        # Apply dehaze to each frame
        dehazed_frame = dehaze(frame, omega=0.5, tmin=0.05, gamma=1.5, color_balance=True)
        dehazed_frame_resized = cv2.resize(dehazed_frame, (resize_width, resize_height))

        cv2.imshow("Beforeupdated", frame_resized)
        cv2.imshow("Afterupdated", dehazed_frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
