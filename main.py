def simple_tokenizer(txt):
    return txt.split()

from torch.serialization import add_safe_globals
add_safe_globals([simple_tokenizer])

from assistance.core import JarvisAssistant
from assistance.utils import wish_me

def main():
    wish_me()
    assistant = JarvisAssistant()
    assistant.run()

if __name__ == "__main__":
    main()