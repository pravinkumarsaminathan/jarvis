import sys
import torch

from .nlp import ChatbotNLP
from .whatsapp import WhatsAppHandler
from .utils import speak, open_app, close_app, schedule, condition, volume_control, command, search_wikipedia, get_weather, play_youtube_music, order_product, get_speed_inference, get_ip_address, nmap_scan_inference, nmap_vulnerable_scan

class JarvisAssistant:
    """
    Main assistant class that manages the chatbot and WhatsApp modules,
    and routes user queries to the appropriate handler.
    """
    def __init__(self):
        self.chatbot = ChatbotNLP()
        self.vectorizer = self.chatbot.vectorizer
        self.model = self.chatbot.model
        self.y_labels = self.chatbot.y_labels

    def run(self, query):
        

            # ---------------- NLP (rule based) ----------------
        if "send" in query and "via whatsapp" in query:
            handler = WhatsAppHandler()
            response = handler.handle(query)
            speak(response)
            print("Bot:", response)
        elif "team name" in query.lower():
            speak("Our Team name is Ninjas")
            print("Bot: Our Team name is Ninjas")
            return
        elif "your name" in query.lower():
            speak("My name is Jarvis , How can i help you today")
            print("Bot: My name is Jarvis , How can i help you today")
            return
        elif "weather" in query.lower():
            response = get_weather(query)
            speak(response)
            print("Bot:", response)
            return
        elif "scan vulnerable ports" in query.lower():
            target_ip = "127.0.0.1"  # Or ask via voice
            response = nmap_vulnerable_scan(target_ip)
            speak(response)
            print("Bot:", response)
            return
        elif "scan" in query.lower() or "ports" in query.lower():
            target_ip = "127.0.0.1"  # Or get IP from user dynamically
            response = nmap_scan_inference(target_ip)
            speak(response)
            print("Bot:", response)
            return
        elif "internet speed" in query.lower() or "network speed" in query.lower():
            speak("Checking internet speed, please wait a moment.")
            response = get_speed_inference()  # Calls the function we defined earlier
            speak(response)  # Your TTS function
            print("Bot:", response)
            return
        elif "my ip" in query.lower() or "ip address" in query.lower():
            response = get_ip_address()
            speak(response)
            print("Bot:", response)
            return
        elif any(query.lower().startswith(w) for w in ["what", "who", "when", "where"]) or "wikipedia" in query.lower():
            response = search_wikipedia(query.replace("on wikipedia", "").strip())
            speak(response)
            print("Bot:", response)
            return
        elif "play" in query.lower() and ("music" in query.lower() or "song" in query.lower()):
            song_name = query.lower().replace("play", "").replace("music", "").strip()
            response = f"Playing {song_name}"
            speak(response)
            print("Bot:", response)
            play_youtube_music(song_name)
            return
        elif any(word in query.lower() for word in ["buy", "purchase", "order"]):
            response = order_product(query)
            speak(response)
            print("Bot:", response)
            return
        

        # ---------------- Normal Chatbot ----------------
        response = self.chatbot.reply(query)

        # Predict intent tag
        X = self.vectorizer.transform([query]).toarray()
        X = torch.tensor(X, dtype=torch.float32)
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.y_labels[predicted.item()]

        # ---------------- Tag Handling ----------------
        if tag == "open_app_control" or tag=="open_app_control_tanglish":
            open_app(query, response)
        elif tag == "close_app_control" or tag=="close_app_control_tanglish":
            close_app(query, response)
        elif "schedule" in query or "university time table" in query:
            schedule()
        elif tag == "system_conditions" or tag=="system_conditions_tanglish":
            if "volume up" in query:
                volume_control("up")
            elif "volume down" in query:
                volume_control("down")
            elif "mute" in query:
                volume_control("mute")
            elif "condition" in query or "battery" in query or "CPU" in query:
                condition()
        else:
            speak(response)
            print("Bot:", response)
