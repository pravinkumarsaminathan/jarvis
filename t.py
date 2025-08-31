from pynput import keyboard

def on_press(key):
    print('Key pressed:', key)

listener = keyboard.Listener(on_press=on_press)
listener.start()
listener.join()