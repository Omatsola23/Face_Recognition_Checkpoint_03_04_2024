import cv2
import streamlit as st
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml')
def detect_faces(rectangle_color,scalefactor,neighbour):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        #Add button to save image
        image="detected_faces.jpg"
        cv2.imwrite(image, frame)
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scalefactor, minNeighbors=neighbour)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            # Convert the rectangle color to BGR format
            bgr_color = tuple(int(rectangle_color[i:i + 2], 16) for i in (1, 3, 5))
            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
def app():
    st.title("Face Detection App")
    st.write("Welcome to the Face Detection App! Follow the instructions below to use the app:")
    # Add instruction
    st.markdown("Adjust the parameters below to customize face detection:")
    st.markdown("   - **Rectangle Color:** Choose the color of the rectangles drawn around detected faces.")
    st.markdown("   - **Min Neighbors:** Adjust the minNeighbors parameter.")
    st.markdown("   - **Scale Factor:** Adjust the scaleFactor parameter.")
    st.markdown("Click the 'Detect Faces' button to start face detection.")
    # Parameters for face detection
    rectangle_color = st.color_picker("Rectangle Color", "#ff0000", key="rectangle_color")
    neighbour=st.slider("Min Neighbors", min_value=1, max_value=10, value=5)
    scalefactor=st.slider("Scale Factor", min_value=1.01, max_value=2.0, step=0.01, value=1.3)
    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Call the detect_faces function with the chosen parameters
        detect_faces(rectangle_color,scalefactor,neighbour)
if __name__ == "__main__":
    app()
