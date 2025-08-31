# Jarvis GUI Application

This project is a voice-activated assistant overlay built using PySide6. It features a visually appealing interface with glassmorphism effects, a pulsing perimeter glow, and an audio visualizer.

## Project Structure

```
jarvis_gui_app
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── app.py
│   ├── bridge.py
│   ├── listeners.py
│   ├── resources
│   │   └── __init__.py
│   ├── widgets
│   │   ├── __init__.py
│   │   ├── glass_panel.py
│   │   ├── perimeter_glow.py
│   │   ├── equalizer.py
│   │   ├── jarvis_overlay.py
│   │   └── click_catcher.py
├── requirements.txt
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd jarvis_gui_app
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command:

```bash
python src/main.py
```

Once the application is running, you can toggle the overlay by pressing `CTRL` twice quickly. Use `CTRL + S` to start listening for voice commands.

## Dependencies

This project requires the following Python packages:

- PySide6
- pynput
- numpy
- sounddevice

You can install these dependencies using the `requirements.txt` file provided in the project.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.