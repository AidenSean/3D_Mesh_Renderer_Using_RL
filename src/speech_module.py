import speech_recognition as sr
import sys

class SpeechProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone_available = True
        try:
            import pyaudio
            # Check if any mic is available
            if not sr.Microphone.list_microphone_names():
                print("No microphone detected.")
                self.microphone_available = False
        except ImportError:
            print("PyAudio not installed. Microphone access unavailable.")
            print("To use speech, please install portaudio (brew install portaudio) and then pip install pyaudio.")
            self.microphone_available = False
        except Exception as e:
            print(f"Microphone error: {e}")
            self.microphone_available = False

    def listen_for_command(self):
        """Listens to the microphone and returns the recognized text. Falls back to input() if no mic."""
        if self.microphone_available:
            try:
                with sr.Microphone() as source:
                    print("\nListening... Say something!")
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    try:
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                        print("Processing audio...")
                        text = self.recognizer.recognize_google(audio)
                        print(f"You said: {text}")
                        return text
                    except sr.WaitTimeoutError:
                        print("Listening timed out.")
                    except sr.UnknownValueError:
                        print("Could not understand audio.")
                    except sr.RequestError as e:
                        print(f"Could not request results; {e}")
            except Exception as e:
                print(f"Error accessing microphone: {e}")
        
        # Fallback
        print("\nMicrophone not available or failed. Please type your prompt:")
        return input(">> ")
