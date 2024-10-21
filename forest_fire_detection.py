import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load your CNN model
model = load_model(r'C:\Users\gaura\Downloads\transfer_learned_model.h5')

# Function to preprocess the frame
def preprocess_frame(frame, target_size=(224, 224)):
    # Resize the frame to match the model's input size
    frame = cv2.resize(frame, target_size)
    # Convert the frame to an array
    frame = img_to_array(frame)
    # Normalize the pixel values (assuming the model was trained with normalized data)
    frame = frame / 255.0
    # Add a batch dimension (required by Keras models)
    frame = np.expand_dims(frame, axis=0)
    return frame

# Open the video file
#video_path = r"C:\Users\gaura\Downloads\Wildfire Forest Fire (FREE STOCK VIDEO).mp4"
video_path = r"C:\Users\gaura\Downloads\Drone Flying Over Forest.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video

    # Preprocess the frame for the model
    processed_frame = preprocess_frame(frame)

    # Make a prediction
    prediction = model.predict(processed_frame)[0]

    # Assuming your model has binary output (0 for no fire, 1 for fire)
    if prediction[1] > 0.6:
        label = "Fire Detected"
        color = (0, 0, 255)  # Red for fire
    else:
        label = "No Fire"
        color = (0, 255, 0)  # Green for no fire

    # Display the result on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the frame
    cv2.imshow('Forest Fire Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
