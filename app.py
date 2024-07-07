from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# Load the model
model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/1')

# Assuming average human height is 170 cm for conversion
AVERAGE_HEIGHT_CM = 170

def process_image(image):
    # Resize and normalize the image to match the model's expected input
    img = cv2.resize(image, (192, 192))
    img = img / 255.0  # Normalize the image
    img = img * 255.0  # Convert back to original range for int32 conversion
    img = img.astype(np.int32)  # Ensure dtype is int32
    img = np.expand_dims(img, axis=0)

    # Debug logging to verify image preprocessing
    print(f"Processed Image Shape: {img.shape}, Dtype: {img.dtype}")

    # Run the model
    result = model.signatures['serving_default'](tf.constant(img))

    # The result will be a dictionary with the model's output
    keypoints = result['output_0'].numpy()[0][0]

    # Debug logging to verify keypoints
    print(f"Keypoints: {keypoints}")

    measurements = calculate_measurements(keypoints)
    return measurements

def calculate_measurements(keypoints):
    # Keypoints indices according to MoveNet's output
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # Calculate distances
    shoulder_width = distance(keypoints[5][:2], keypoints[6][:2])
    arm_length_left = distance(keypoints[5][:2], keypoints[7][:2]) + distance(keypoints[7][:2], keypoints[9][:2])
    arm_length_right = distance(keypoints[6][:2], keypoints[8][:2]) + distance(keypoints[8][:2], keypoints[10][:2])
    leg_length_left = distance(keypoints[11][:2], keypoints[13][:2]) + distance(keypoints[13][:2], keypoints[15][:2])
    leg_length_right = distance(keypoints[12][:2], keypoints[14][:2]) + distance(keypoints[14][:2], keypoints[16][:2])
    waist = distance(keypoints[11][:2], keypoints[12][:2])

    # Convert normalized distances to approximate cm
    # Assuming the distance from top of head (keypoint 0) to bottom of feet (keypoint 15 or 16) is the average height
    height = max(distance(keypoints[0][:2], keypoints[15][:2]), distance(keypoints[0][:2], keypoints[16][:2]))
    scale = AVERAGE_HEIGHT_CM / height

    # Debug logging to verify distances and scale
    print(f"Height: {height}, Scale: {scale}")

    measurements = {
        "shoulder_width_cm": float(shoulder_width * scale),
        "arm_length_left_cm": float(arm_length_left * scale),
        "arm_length_right_cm": float(arm_length_right * scale),
        "leg_length_left_cm": float(leg_length_left * scale),
        "leg_length_right_cm": float(leg_length_right * scale),
        "waist_cm": float(waist * scale)
    }

    return measurements

@app.route('/measure', methods=['POST'])
def measure():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Debug logging to verify image loading
    print(f"Uploaded Image Shape: {img.shape}, Dtype: {img.dtype}")

    measurements = process_image(img)
    return jsonify(measurements)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
