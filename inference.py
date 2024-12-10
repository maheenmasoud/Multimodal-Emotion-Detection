import cv2
import numpy as np
import time
import speech_recognition as sr

def capture_image(camera, count):
    print("Capturing image...")
    ret, frame = camera.read()
    if ret:
        image_path = "captured_image.jpg"
        cv2.imwrite(image_path, frame)
        print(f"Image captured and saved at {image_path}")
        
        return image_path
    else:
        print("Failed to capture image")
        return None

def record_audio(count, camera):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording audio...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            image_path = capture_image(camera, count)
            print(f"Transcription: {text}")
            return text, image_path
        except sr.WaitTimeoutError:
            print("Timeout. No audio detected.")
            return None
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

def predict_emotion_live(image_path, text, model):
    return "Happy"  #placeholder

def live_emotion_detection(interval=5, model=None,):
    count = 0
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open the camera.")
        return
    
    print("Press 'q' to quit.")
    try:
        while True:
            count += 1
            text, image_path = record_audio(count, camera)
            
            if image_path and text:
                emotion = predict_emotion_live(image_path, text, model)
                print(f"Predicted Emotion: {emotion}")
            
            # Wait for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(interval)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()

def main():
    live_emotion_detection()

if __name__ == "__main__":
    main()