import sys
import torch

from .nlp import ChatbotNLP
from .whatsapp import WhatsAppHandler
from .utils import speak, open_app, close_app, schedule, condition, volume_control, command

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
        # #wish_me()
        # while True:
        #     #query = command()
        #     query = input("Enter input:").lower()

        #     if query == "none":
        #         continue
        #     if "exit" in query:
        #         speak("Goodbye Boss!")
        #         sys.exit()

            # ---------------- WhatsApp Handling ----------------
        if "send" in query and "via whatsapp" in query:
            handler = WhatsAppHandler()
            response = handler.handle(query)
            speak(response)
            print("Bot:", response)

        # ---------------- Normal Chatbot ----------------
        response = self.chatbot.reply(query)

        # Predict intent tag
        X = self.vectorizer.transform([query]).toarray()
        X = torch.tensor(X, dtype=torch.float32)
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.y_labels[predicted.item()]

        # ---------------- Tag Handling ----------------
        if tag == "open_app_control":
            open_app(query)
        elif tag == "close_app_control":
            close_app(query)
        elif "schedule" in query or "university time table" in query:
            schedule()
        elif tag == "system_conditions":
            if "volume up" in query:
                volume_control("up")
            elif "volume down" in query:
                volume_control("down")
            elif "mute" in query:
                volume_control("mute")
            elif "condition" in query:
                condition()
            # elif tag == "vulnerable_ports":
            #     scan_vulnerable_ports()
            # elif tag == "close_port":
            #     match = re.search(r'\d+', query)
            #     if match:
            #         port = match.group()
            #         close_port(port)
            #     else:
            #         speak("Please specify which port you want me to close.")
        else:
            speak(response)
            print("Bot:", response)