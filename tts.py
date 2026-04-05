import pyttsx3
import threading
from queue import Queue

speech_queue = Queue()

def _speech_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
        finally:
            speech_queue.task_done()

# Start background thread once
thread = threading.Thread(target=_speech_worker, daemon=True)
thread.start()

def speak(text):
    speech_queue.put(text)