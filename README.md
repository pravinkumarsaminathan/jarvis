# Jarvis – Linux Voice Assistant (Offline, Intent-Based AI)

Jarvis is a Linux-only AI desktop assistant that you can open instantly using keyboard shortcuts.  
It supports **voice + text commands**, uses a **local NLP model** to understand user intent,  
and responds using a **local Edge-TTS server** (open-source).

Built for speed, and full on-device control.

---

## Features

###  Global Popup Launcher
- Double-press **Ctrl** → Jarvis appears anywhere on your screen.
- Type or speak commands instantly.

###  Voice Control
- Hold **Ctrl + S** → speak  
- Release → Jarvis processes your speech  
- Converts speech → text → NLP → action

###  Local NLP Intent Model
Understands commands such as:
- `open_app`
- `search_web`
- `play_media`
- `scan_network`
- `get_info`
- `system_control`
- ...and more.

###  Local TTS (Edge-TTS Server)
- Jarvis speaks responses using a **local Edge-TTS server**.
- Fully offline and open-source.
- Press **Q** anytime to stop the speech output.

###  Linux Only
Supports Linux desktops.

---

##  How to Use

| Action | Shortcut |
|--------|----------|
| Open Jarvis popup | **Ctrl** (twice) |
| Voice command | Hold **Ctrl + S**, speak, release |
| Stop voice speaking | Press **Q** |
| Close popup | click anyware or **Ctrl** (twise) |

### Example Commands
- “Open VS Code”
- “Scan my open port”
- “Who is Elon Musk?”
- “What is my system condition”
- “Open my last workspace”
- “Play song”
---

##  Architecture
Double Ctrl → Popup UI → (Optional: STT) → NLP Intent Model → Skill Router → Linux Action Module → Result (Text + Edge-TTS Voice)

##  Requirements

- Linux (Ubuntu / Arch / Fedora / etc.)
- Python 3.x
- Installed local **Edge-TTS server**

## setup process will be updated soon


