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
from pynput import keyboard
import tempfile
import speedtest
import socket

VULNERABLE_PORTS = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    69: "TFTP",
    80: "HTTP",
    110: "POP3",
    139: "NetBIOS",
    143: "IMAP",
    443: "HTTPS",
    445: "SMB",
    3389: "RDP"
}

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
        
# def detect_voice(query):
#     """
#     Detect if the query is in English or Thanglish (Tamil in Latin script).
#     Returns the appropriate voice string.
#     """
#     words = set(query.lower().split())
#     if THANGLISH_KEYWORDS & words:
#         return "ta-IN-PallaviNeural"
#     return "en-US-AriaNeural"

# _speak_lock = threading.Lock()

# def speak(text, speed=1.0):
#     with _speak_lock:
#         voice = detect_voice(text)
#         """
#         Stream TTS from local server and play instantly with mpg123.
#         """
#         payload = {
#             "input": text,
#             "voice": voice,
#             "response_format": "mp3",
#             "speed": speed
#         }

#         curl_cmd = [
#             "curl", "-s", "-X", "POST", "http://localhost:5050/v1/audio/speech",
#             "-H", "Content-Type: application/json",
#             "-H", "Authorization: Bearer your_api_key_here",
#             "-d", json.dumps(payload)
#         ]

#         mpg123_cmd = ["mpg123", "-"]
#         curl_proc = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE)
#         subprocess.run(mpg123_cmd, stdin=curl_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         curl_proc.stdout.close()
#         curl_proc.wait()
    #os.system(f'edge-playback --text "{text}" --voice en-US-AriaNeural > /dev/null 2>&1')


_speak_lock = threading.Lock()
_stop_speaking = threading.Event()

def speak(text, speed=1.0):
    global _stop_speaking
    with _speak_lock:
        _stop_speaking.clear()
        voice ="en-US-AriaNeural"

        payload = {
            "input": text,
            "voice": voice,
            "response_format": "mp3",
            "speed": speed
        }

        # Save TTS to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_filename = tmp_file.name

        curl_cmd = [
            "curl", "-s", "-X", "POST", "http://localhost:5050/v1/audio/speech",
            "-H", "Content-Type: application/json",
            "-H", "Authorization: Bearer your_api_key_here",
            "-d", json.dumps(payload),
            "-o", tmp_filename  # save output to file
        ]

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        subprocess.run(curl_cmd)

        # Play the file with mpg123
        player_proc = subprocess.Popen(["mpg123", tmp_filename],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        while player_proc.poll() is None:
            if _stop_speaking.is_set():
                player_proc.terminate()
                break

        player_proc.wait()
        os.remove(tmp_filename)  # clean up temp file

# Listener for 'q' key
def on_press(key):
    try:
        if key.char == 'q':
            _stop_speaking.set()
            return False
    except AttributeError:
        pass

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
    elif "youtube" in cmd:
        youtube()
    elif "code" in cmd or "visual studio" in cmd or "v s code" in cmd:
        subprocess.Popen(["code"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        speak(response)
    elif "terminal" in cmd or "command prompt" in cmd or "cmd" in cmd:
        # XFCE terminal
        try:
            speak(response)
            subprocess.Popen(["qterminal"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            speak(response)
        except FileNotFoundError:
            speak("Terminal application not found.")
    elif "file manager" in cmd or "files" in cmd or "explorer" in cmd:
        # XFCE file manager
        try:
            speak(response)
            subprocess.Popen(["thunar"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            speak("File manager application not found.")
    elif "camera" in cmd:
        speak(response)
        subprocess.Popen(['cheese'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    elif "code" in cmd or "visual studio" in cmd or "v s code" in cmd:
        speak(response)
        os.system("pkill -f code > /dev/null 2>&1")
    elif "file manager" in cmd or "files" in cmd or "explorer" in cmd:
        speak(response)
        os.system("pkill -f thunar > /dev/null 2>&1")
    elif "camera" in cmd:
        speak(response)
        os.system("pkill -f cheese > /dev/null 2>&1")
    else:
        speak("sorry i don't have access to do that.")

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

def youtube():
    speak("Boss, what should I search on youtube?")
    s = command()
    if s != "None":
        webbrowser.open(f"https://www.youtube.com/search?q={s}")

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

def play_youtube_music(song_name):
    # Get the first video URL quickly
    url = subprocess.check_output(
        ["yt-dlp", f"ytsearch1:{song_name}", "--get-url", "-f", "bestaudio"],
        text=True  # ensures output is a string, not bytes
    ).strip()

    # Play it instantly in mpv
    #os.system(f'mpv --no-video "{url}"')
    subprocess.Popen([
        "mpv", "--force-window=yes", "--osc=yes", url
    ])
    speak("Playing song....")
    print("playing song....")

def order_product(query: str) -> str:
    """
    Extracts product name from query and searches on Amazon or Flipkart.
    Returns a short response for speech.
    """

    q = query.lower().strip()

    # Detect platform
    if "flipkart" in q:
        platform = "Flipkart"
    elif "amazon" in q:
        platform = "Amazon"
    else:
        platform = "Amazon"  # default

    # Remove platform words
    q = q.replace("on amazon", "").replace("on flipkart", "").strip()

    # Extract product: everything after buy/purchase/order
    product = None
    for keyword in ["buy", "purchase", "order"]:
        if keyword in q:
            product = q.split(keyword, 1)[1].strip()
            break

    if not product:
        return "What product would you like me to order?"

    # Platform-specific search
    if platform == "Amazon":
        webbrowser.open(f"https://www.amazon.in/s?k={product.replace(' ', '+')}")
    elif platform == "Flipkart":
        webbrowser.open(f"https://www.flipkart.com/search?q={product.replace(' ', '+')}")

    return f"Searching deals on {product} on {platform}."

def get_speed_inference():
    st = speedtest.Speedtest()
    st.get_best_server()
    download_speed = st.download() / 1_000_000  # Mbps
    upload_speed = st.upload() / 1_000_000      # Mbps
    ping = st.results.ping

    # Make inference based on speed
    if download_speed > 50:
        download_status = "excellent"
    elif download_speed > 20:
        download_status = "good"
    else:
        download_status = "slow"

    if upload_speed > 20:
        upload_status = "excellent"
    elif upload_speed > 10:
        upload_status = "good"
    else:
        upload_status = "slow"

    inference = (f"Your internet speed is as follows: Download is {download_speed:.2f} Mbps, "
                 f"which is {download_status}. Upload is {upload_speed:.2f} Mbps, "
                 f"which is {upload_status}. Ping is {ping} ms.")

    return inference

def get_ip_address():
    # Get local IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    # # Get public IP
    # try:
    #     public_ip = requests.get('https://api.ipify.org').text
    # except:
    #     public_ip = "Could not fetch public IP. Check your internet connection."

    return f"Your local IP address is {local_ip}."# Your public IP address is {public_ip}."


def nmap_scan_inference(target_ip):
    """
    Performs a basic TCP scan and returns a human-readable inference.
    """
    try:
        # Run Nmap TCP scan
        result = subprocess.run(["nmap", "-sT", target_ip], capture_output=True, text=True)
        output = result.stdout

        # Simple inference: find open ports
        open_ports = []
        for line in output.splitlines():
            if "open" in line:
                parts = line.split()
                port = parts[0] if len(parts) > 0 else ""
                service = parts[2] if len(parts) > 2 else ""
                open_ports.append(f"{port} ({service})")

        # Prepare AI-style response
        if open_ports:
            inference = f"I scanned this device and found the following open ports: {', '.join(open_ports)}."
        else:
            inference = f"I scanned this device but could not find any open ports."

        return inference

    except FileNotFoundError:
        return "Nmap is not installed. Please install Nmap and try again."
    except Exception as e:
        return f"Error running Nmap scan: {e}"
    

def nmap_vulnerable_scan(target_ip):
    """
    Scans target IP and reports vulnerable/risky ports found.
    """
    try:
        # Basic TCP scan
        result = subprocess.run(["nmap", "-sT", target_ip], capture_output=True, text=True)
        output = result.stdout

        # Find open ports
        open_vulnerable_ports = []
        for line in output.splitlines():
            if "open" in line:
                parts = line.split()
                port_num = int(parts[0].split("/")[0])
                service = VULNERABLE_PORTS.get(port_num, parts[2] if len(parts) > 2 else "Unknown")
                if port_num in VULNERABLE_PORTS:
                    open_vulnerable_ports.append(f"{port_num} ({service})")

        # Prepare inference
        if open_vulnerable_ports:
            inference = (f"Warning: I found potentially vulnerable ports on this device: "
                         f"{', '.join(open_vulnerable_ports)}. "
                         "It is recommended to secure these ports.")
        else:
            inference = f"No commonly vulnerable ports found on this device. Your system looks safer."

        return inference

    except FileNotFoundError:
        return "Nmap is not installed. Please install Nmap and try again."
    except Exception as e:
        return f"Error running Nmap scan: {e}"
    
