# import necessary libaries
import streamlit as st
import numpy as np
import cv2
from keras.models import model_from_json
from PIL import Image

# Load model
def load_model():
    json_file = open('model/EmoLearn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("model/EmoLearn_model.h5")
    return emotion_model

# predict face emotion from Images
def predict_emotions_image(image_array, emotion_model):
    
    # Create face detector
    face_detector = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')

    # Convert the frame to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Detect faces in the frame    
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray_frame = gray_image[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[maxindex]

        return emotion_label

# predict face emotion from Webcam
def predict_emotions_webcam(frame, emotion_model):
    # Create face detector
    face_detector = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[maxindex]

        # Draw the bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)      
    return frame

# Define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Create the Streamlit app
def main():
    # Show Logo image
    st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
    )
    image = Image.open('Logo.png')
    st.sidebar.image(image, caption='EmoLearn')
    # Title 
    st.markdown("<h1 style='text-align: center;'>Emo<span style='color: blue;'>Learn</span></h1>", unsafe_allow_html=True) 
    # Choose options: Upload Image or Webcam
    inputResource = st.sidebar.selectbox('How would you like to be detected?', ['select here...', 'Image', 'Webcam'])

    # Load the emotion detection model
    emotion_model = load_model()

    # HomePage of Streamlit App
    if inputResource == 'select here...':
        st.markdown("<h4 style='text-align: center;'>Emotion Detection from Images and Webcam</h4><br><br>", unsafe_allow_html=True) 
        # Cababilities 
        st.markdown("<h5 style='text-align: left;'>Cababilities!</h5>", unsafe_allow_html=True) 
        st.write('- Accessible to everyone, regardless of their race, gender and black or white.')
        st.write('- Don\'t save or capture any input data from users.')
        st.write('')
        # Limitations
        st.markdown("<h5 style='text-align: left;'>Limitations!</h5>", unsafe_allow_html=True) 
        st.write('- Only grayscale images can be predicted.')
        st.write('- May vary depending on lighting conditions, camera quality, and camera angle.')

    elif inputResource == 'Image':
        # Clear the main area
        st.empty()
        # Title of Option 'Image'
        st.markdown("<h4 style='text-align: center;'>Emotion Detection from Images</h4><br><br>", unsafe_allow_html=True) 
        # File uploader allows users to upload images
        uploaded_file = st.file_uploader('Upload your image file here...', type=['jpeg', 'jpg', 'png'])
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, width=300)
            try:
                # Read the uploaded image using cv2.imread()
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Predict the emotion of the grayscale image
                emotion_label = predict_emotions_image(image_array, emotion_model)

                # Display the predicted emotion 
                st.markdown("<span style='text-align: center; font-size: 14'>Predicted Emotion : </span>"
                            f"<span style='text-align: center; color: blue; font-size: 14'>{emotion_label}</span>", unsafe_allow_html=True)
                                
                cv2.destroyAllWindows() 
            except Exception as e:
                st.write(str(e))
                st.error("Error occurred while processing the image. Please make sure the image format is supported (JPEG, JPG, PNG) and try again.")
        
    elif inputResource == 'Webcam':
        # Clear the main area
        st.empty()
        # Title of Option 'Webcam'
        st.markdown("<h4 style='text-align: center;'>Emotion Detection from Webcam</h4><br><br>", 
                    unsafe_allow_html=True) 
        # Start the webcam feed
        cap = cv2.VideoCapture(0)

        # Display the webcam feed in the main body
        video_placeholder = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Detect emotion
            processed_frame = predict_emotions_webcam(frame, emotion_model)
            # Show bounding box
            video_placeholder.image(processed_frame, channels='BGR')

        cap.release()
        cv2.destroyAllWindows()

# Load the Streamlit app
if __name__ == '__main__':
    main()
