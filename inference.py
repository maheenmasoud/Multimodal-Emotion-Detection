# live_emotion_inference.py

import cv2
import numpy as np
import time
import speech_recognition as sr
from gtts import gTTS
import os
import sys  # Needed for checking platform
import torch
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
import shutil

# Import models and transformations from dl_final_project.py
from dl_final_project import MultimodalModel, SquarePad

# ---------------------------
# Preprocessing Functions
# ---------------------------

def preprocess_image(image_path):
    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((128, 128)),

        # No data augmentation during inference
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    image = transform(image)
    return image

def preprocess_text(text, tokenizer, max_length=50):
    encoded = tokenizer(
        text,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)


def extract_and_save_face(img_path, output_dir, face_cascade):
    """
    Extracts the largest face from an image and saves it to the output directory.
    If no face is found, the original image is copied to the output directory.
    
    Parameters:
        img_path (str): Path to the input image.
        output_dir (str): Directory where the processed image will be saved.
        face_cascade (cv2.CascadeClassifier): Haar Cascade classifier for face detection.
        
    Returns:
        None
    """
    filename = os.path.basename(img_path)
    
    # Read the image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        return
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        # No faces found; copy the original image to the output directory
        output_path = os.path.join(output_dir, filename)
        shutil.copy(img_path, output_path)
        print(f"No faces found in '{filename}'. Copied the original image.")
        return
    
    # Select the largest face based on area (width * height)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # rect = (x, y, w, h)
    x, y, w, h = largest_face
    face = image[y:y+h, x:x+w]
    
    # Save the extracted face with the same filename
    face_path = os.path.join(output_dir, filename)
    cv2.imwrite(face_path, face)
    print(f"Saved the largest face from '{filename}' to '{face_path}'.")


# ---------------------------
# Prediction Function
# ---------------------------

def predict_emotion_live(image_path, text, model, tokenizer, device, label_map):
    model.eval()
    with torch.no_grad():
        # Preprocess image
        image = preprocess_image(image_path)
        if image is None:
            return None
        image = image.unsqueeze(0).to(device)  # Add batch dimension

        # Preprocess text
        input_ids, attention_mask = preprocess_text(text, tokenizer)
        input_ids = input_ids.unsqueeze(0).to(device)        # Add batch dimension
        attention_mask = attention_mask.unsqueeze(0).to(device)  # Add batch dimension

        # Forward pass
        outputs = model(input_ids, attention_mask, image)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

        # Map prediction to label
        predicted_label = label_map[predicted.item()]
        confidence = probabilities[0][predicted.item()].item()

    return predicted_label, confidence

# ---------------------------
# Result Output Function
# ---------------------------

def output_result(emotion):
    language = 'en'
    if emotion:
        tts = gTTS(text=f"You are feeling {emotion}", lang=language, slow=False)
        tts.save("output.mp3")
        print(f"The predicted emotion is {emotion}")
    else:
        print("No emotion detected.")
        tts = gTTS(text="No emotion detected.", lang=language, slow=False)
        tts.save("output.mp3")

    # Play the audio file
    if os.name == 'nt':  # For Windows
        os.system("start output.mp3")
    elif os.name == 'posix':  # For macOS and Linux
        if sys.platform == "darwin":  # macOS
            os.system("afplay output.mp3")
        else:  # Linux
            os.system("mpg123 output.mp3")  # Ensure mpg123 is installed
    else:
        print("Unsupported OS for audio playback.")

# ---------------------------
# Live Emotion Detection
# ---------------------------

def live_emotion_detection(cascade_path, face_cascade, interval=5, model=None, tokenizer=None, device=None, label_map=None):
    if model is None or tokenizer is None or device is None or label_map is None:
        print("Model, tokenizer, device, and label_map must be provided.")
        return

    count = 0
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open the camera.")
        return

    recognizer = sr.Recognizer()

    print("Press 'q' to quit.")
    try:
        while True:
            count += 1

            # Record audio
            text, audio_success = record_audio(recognizer, camera)
            if text is None:
                emotion = None
                output_result(emotion)
                time.sleep(interval)
                continue

            # Capture image
            image_path = capture_image(camera, count, cascade_path, face_cascade)
            if image_path is None:
                emotion = None
                output_result(emotion)
                time.sleep(interval)
                continue

            # Predict emotion
            prediction = predict_emotion_live(image_path, text, model, tokenizer, device, label_map)
            if prediction is not None:
                emotion, confidence = prediction
                print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2f})")
            else:
                emotion = None

            # Output the result
            output_result(emotion)

            # Wait for the specified interval
            time.sleep(interval)

            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()

# ---------------------------
# Audio and Image Capture Functions
# ---------------------------

def capture_image(camera, count, cascade_path, face_cascade):
    print("Capturing image...")

    # Verify if the cascade was loaded successfully
    if face_cascade.empty():
        print(f"Error: Could not load Haar Cascade XML file from {cascade_path}")
        sys.exit(1)  # Exit the script with an error code

    ret, image = camera.read()
    if ret:
        image_path = f"captured_image_{count}.jpg"

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            # No faces found; copy the original image to the output directory
            print(f"No faces found in '{image_path}'. Copied the original image.")
            processed_image = image
        else:
            # Select the largest face based on area (width * height)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # rect = (x, y, w, h)
            x, y, w, h = largest_face
            processed_image = image[y:y+h, x:x+w]
            
        
        cv2.imwrite(image_path, processed_image)
        print(f"Image captured and saved at {image_path}")
        return image_path
    else:
        print("Failed to capture image")
        return None

def record_audio(recognizer, camera):
    with sr.Microphone() as source:
        print("Recording audio...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            print(f"Transcription: {text}")
            return text, True
        except sr.WaitTimeoutError:
            print("Timeout. No audio detected.")
            return None, False
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return None, False
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None, False

# ---------------------------
# Main Function
# ---------------------------

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Initialize the model
    model = MultimodalModel()
    model.to(device)

    # Load the trained weights
    model_path = 'best_model.pth'  # Ensure this path is correct
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    model.eval()

    # Define label mapping
    label_map = {0: 'neutral', 1: 'joy', 2: 'sadness', 3: 'fear', 4: 'anger', 5: 'surprise', 6: 'disgust'}

    # Load the Haar Cascade classifier for face detection using OpenCV's built-in path
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Start live emotion detection
    live_emotion_detection(cascade_path, face_cascade, interval=5, model=model, tokenizer=tokenizer, device=device, label_map=label_map)

if __name__ == "__main__":
    main()
