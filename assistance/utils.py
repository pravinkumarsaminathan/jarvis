import datetime
import os
import time
import webbrowser
import subprocess
import psutil
import json
import requests
import speech_recognition as sr
import nmap
import threading
import wikipedia

OPENWEATHER_API_KEY = "ca8004bcd630a58b8fc39755e6dfb46c"   # replace with your API key

THANGLISH_KEYWORDS = {
    "hi", "da", "saptiya", "eppadi", "iruka", "vanakam", "bro", "hey", "macha", "enna", "pandra", "hello",
    "bye", "pakkalam", "seri", "poi", "varen", "romba", "help", "pannita", "super", "oru",
    "joke", "sollu", "comedy", "sirikkava", "nee", "yaru", "innaiki", "naal", "day", "news",
    "sup", "haha", "lol", "semma", "sirippu", "unna", "create", "pannanga",
    "panra", "seiya", "pesitu", "iruken", "vayasu", "eppo", "aana",
    "pannu", "la", "po", "chrome", "close", "exit", "quit", "kaatu", "panniten", "pannittu",
    "panren", "pesina", "nalla", "pesuna", "vendam", "iruken", "unakku",
    "un", "seekiram", "pannalaam", "iruka", "santhosham", "mathiri"
}

def command():
    r = sr.Recognizer()
    # Suppress ALSA/JACK errors
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening...")
            audio = r.listen(source, phrase_time_limit=10)
        try:
            print("\rRecognition...")
            query = r.recognize_google(audio, language="en-in")
            print(f"\rUser said: {query}")
        except Exception:
            print("Say that again please...")
            return "none"
        return query.lower()
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)

def detect_voice(query):
    """
    Detect if the query is in English or Thanglish (Tamil in Latin script).
    Returns the appropriate voice string.
    """
    words = set(query.lower().split())
    if THANGLISH_KEYWORDS & words:
        return "ta-IN-PallaviNeural"
    return "en-US-AriaNeural"

_speak_lock = threading.Lock()

def speak(text, speed=1.0):
    with _speak_lock:
        voice = detect_voice(text)
        """
        Stream TTS from local server and play instantly with mpg123.
        """
        payload = {
            "input": text,
            "voice": voice,
            "response_format": "mp3",
            "speed": speed
        }

        curl_cmd = [
            "curl", "-s", "-X", "POST", "http://localhost:5050/v1/audio/speech",
            "-H", "Content-Type: application/json",
            "-H", "Authorization: Bearer your_api_key_here",
            "-d", json.dumps(payload)
        ]

        mpg123_cmd = ["mpg123", "-"]
        curl_proc = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE)
        subprocess.run(mpg123_cmd, stdin=curl_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        curl_proc.stdout.close()
        curl_proc.wait()
    #os.system(f'edge-playback --text "{text}" --voice en-US-AriaNeural > /dev/null 2>&1')

def wish_me():
    hour = int(datetime.datetime.now().hour)
    t = time.strftime("%I:%M %p")  # ✅ gives "12:00 AM"
    day = cal_day()
    if hour < 12:
        speak(f"Good morning Pravin, it is {day} and the time is {t}")
    elif hour < 17:
        speak(f"Good afternoon Pravin, it is {day} and the time is {t}")
    else:
        speak(f"Good evening Pravin, it is {day} and the time is {t}")

def cal_day():
    day = datetime.datetime.today().weekday() + 1
    return ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][day-1]

def open_app(cmd, response):
    if "calculator" in cmd:
        subprocess.Popen(["gnome-calculator"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        speak(response)
    elif "notepad" in cmd:
        subprocess.Popen(["gedit"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        speak(response)
    elif "paint" in cmd:
        subprocess.Popen(["gimp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        speak(response)
    elif "google" in cmd:
        browsing()
    else:
        sites = {
            "facebook": "https://www.facebook.com/",
            "whatsapp": "https://web.whatsapp.com/",
            "discord": "https://discord.com/",
            "instagram": "https://www.instagram.com/"
        }
        for key, url in sites.items():
            if key in cmd:
                speak(f"Opening your {key}")
                webbrowser.open(url)
                return
        speak("No result found")

def close_app(cmd, response):
    if "calculator" in cmd:
        speak(response)
        os.system("pkill -f gnome-calculator > /dev/null 2>&1")
    elif "notepad" in cmd:
        speak(response)
        os.system("pkill -f gedit > /dev/null 2>&1")
    elif "paint" in cmd:
        speak(response)
        os.system("pkill -f gimp > /dev/null 2>&1")
    else:
        speak("sorry i don't have access to do that.")

# def scan_vulnerable_ports():
#     try:
#         nm = nmap.PortScanner()
#         # Run nmap with vulnerability detection scripts
#         nm.scan('127.0.0.1', arguments='-sV --script vuln')
#         speak("Scanning for vulnerable ports...")
#         vulnerable_ports = []

#         for host in nm.all_hosts():
#             for proto in nm[host].all_protocols():
#                 for port, details in nm[host][proto].items():
#                     if details['state'] == "open":
#                         script_output = details.get('script', {})
#                         if script_output:  # If vulnerability script found something
#                             vulnerable_ports.append(f"{port} ({details['name']})")
#         speak("Finished scanning...")
#         if vulnerable_ports:
#             ports_str = ", ".join(vulnerable_ports)
#             speak(f"The vulnerable ports on your system are {ports_str}")
#         else:
#             speak("No vulnerable ports detected on your system")
#     except Exception as e:
#         speak("Sorry boss, I could not complete the port scan")
#         print(f"[Error] {e}")


# def close_port(port):
#     try:
#         # Using ufw to block the port
#         cmd = f"sudo ufw deny {port}"
#         os.system(cmd + " > /dev/null 2>&1")
#         speak(f"Port {port} has been blocked successfully")
#     except Exception as e:
#         speak(f"Sorry boss, I could not close port {port}")
#         print(f"[Error] {e}")


def schedule():
    today = cal_day().lower()
    week = {
        "monday": "Boss, Monday starts strong! 9–9:50 Algorithms, then 10–11:50 System Design. Lunch at 12, and 2 PM Programming Lab.",
        "tuesday": "Boss, Tuesday’s packed! 9 Web Dev, 11 DB Systems. Lunch at 1, then Open Source Lab at 2.",
        "wednesday": "Boss, Wednesday vibes! 9 ML, 11 OS, 12 Ethics. Break at 1, and 2 PM Software Engg Workshop.",
        "thursday": "Boss, Thursday flow! 9 Networks, 11 Cloud Computing. Lunch at 1, then Cybersecurity Lab at 2.",
        "friday": "Boss, it’s Friday grind! 9 AI, 10 Adv Programming, 11 UI/UX. Lunch at 1, and Capstone work starts at 2.",
        "saturday": "Boss, Saturday schedule: morning Capstone meetings, noon Innovation session, and coding practice.",
        "sunday": "Boss, Sunday is chill mode! Rest or catch up on work."
    }
    speak(week.get(today, "No schedule found"), speed=0.8)

def social_media(cmd):
    sites = {
        "facebook": "https://www.facebook.com/",
        "whatsapp": "https://web.whatsapp.com/",
        "discord": "https://discord.com/",
        "instagram": "https://www.instagram.com/"
    }
    for key, url in sites.items():
        if key in cmd:
            speak(f"Opening your {key}")
            webbrowser.open(url)
            return
    speak("No result found")

def browsing():
    speak("Boss, what should I search on Google?")
    s = command()
    if s != "None":
        webbrowser.open(f"https://www.google.com/search?q={s}")

def condition():
    usage = str(psutil.cpu_percent())
    speak(f"CPU is at {usage} percent")
    battery = psutil.sensors_battery()
    if battery:
        speak(f"Our system has {int(battery.percent)} percent battery")
    else:
        speak("I can't detect battery on this system")

def volume_control(action):
    if action == "up":
        os.system("pactl set-sink-volume @DEFAULT_SINK@ +5% > /dev/null 2>&1")
        speak("Volume increased")
    elif action == "down":
        os.system("pactl set-sink-volume @DEFAULT_SINK@ -5% > /dev/null 2>&1")
        speak("Volume decreased")

def search_wikipedia(query):
    try:
        # Search Wikipedia summary (first 2 sentences)
        result = wikipedia.summary(query, sentences=2)
        print(f"\n Wikipedia Result for '{query}':\n{result}\n")
        speak(result)
    except wikipedia.exceptions.DisambiguationError as e:
        speak("Your query is too broad, please be more specific.")
        print("Disambiguation Error:", e.options[:5])  # show some options
    except wikipedia.exceptions.PageError:
        speak("Sorry, I couldn't find anything on Wikipedia for that.")
    except Exception as e:
        speak("An error occurred while searching Wikipedia.")
        print("Error:", e)

def get_weather(query: str) -> str:
    city = extract_city(query)
    if not city:
        return "Please specify a city, like 'weather in chennai today'."

    # detect <when>
    if "tomorrow" in query:
        when = "tomorrow"
    elif "next" in query:
        when = "forecast"
    else:
        when = "today"

    try:
        if when == "today":
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
            data = requests.get(url).json()
            if data.get("cod") != 200:
                return f"Couldn't find weather for {city}."
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"Today in {city.title()}: {desc}, {temp}°C."

        else:  # tomorrow or forecast
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
            data = requests.get(url).json()
            if data.get("cod") != "200":
                return f"Couldn't get forecast for {city}."

            forecasts = []
            now = datetime.datetime.now()

            for entry in data["list"]:
                date_txt = entry["dt_txt"]
                forecast_date = datetime.datetime.strptime(date_txt, "%Y-%m-%d %H:%M:%S")

                if when == "tomorrow" and forecast_date.date() == (now + datetime.timedelta(days=1)).date():
                    temp = entry["main"]["temp"]
                    desc = entry["weather"][0]["description"]
                    forecasts.append(f"At {forecast_date.strftime('%I %p')}: {desc}, {temp}°C")

                elif when == "forecast" and forecast_date.hour == 12:
                    temp = entry["main"]["temp"]
                    desc = entry["weather"][0]["description"]
                    forecasts.append(f"{forecast_date.strftime('%A')}: {desc}, {temp}°C")

            if forecasts:
                return f"Weather in {city.title()} ({when}):\n" + "\n".join(forecasts)
            else:
                return f"No forecast available for {city}."

    except Exception as e:
        return f"Error while fetching weather: {str(e)}"
    
def extract_city(query: str) -> str:
    """
    Extract city name from a query like:
    'weather in chennai today' -> 'chennai'
    """
    if " in " in query:
        city = query.split(" in ", 1)[1]  # take text after 'in'
        # remove timing words
        for w in ["today", "tomorrow", "next", "days", "day"]:
            city = city.replace(w, "").strip()
        return city
    return None
