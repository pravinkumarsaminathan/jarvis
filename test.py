import re
# ...existing code...

def parse_send_message_command(cmd):
    """
    Extracts name, message, and via from commands like:
    'hey jarvis send good morning message to john via whatsapp'
    Supports: whatsapp, facebook, instagram, telegram, etc.
    Returns a dict: {'name': ..., 'message': ..., 'via': ...}
    """
    cmd = cmd.lower()
    # Try to extract 'via' (e.g., via whatsapp, via facebook, etc.)
    via_match = re.search(r'via (\w+)', cmd)
    via = via_match.group(1) if via_match else None

    # Try to extract name (e.g., to john, to alice, to mom)
    name_match = re.search(r'to ([a-z]+)', cmd)
    name = name_match.group(1) if name_match else None

    # Try to extract message (between 'send' and 'message to' or 'to')
    msg_match = re.search(r'send (.+?) message to', cmd)
    if not msg_match:
        msg_match = re.search(r'send (.+?) to', cmd)
    message = msg_match.group(1).strip() if msg_match else None

    return {'name': name, 'message': message, 'via': via}

# Example usage:
cmd = "hey jarvis send good morning message to john via whatsapp"
result = parse_send_message_command(cmd)
print(result)  # {'name': 'john', 'message': 'good morning', 'via': 'whatsapp'}

cmd2 = "send hello to alice on telegram"
result2 = parse_send_message_command(cmd2)
print(result2)  # {'name': 'alice', 'message': 'hello', 'via': None}

# ...existing code...