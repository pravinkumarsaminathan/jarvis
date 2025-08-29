import asyncio
import edge_tts
import subprocess

async def main():
    text = "Hello, this is Edge TTS streaming MP3 immediately!"
    tts = edge_tts.Communicate(text, "en-US-AriaNeural")

    # Start mpg123 (reads MP3 from stdin)
    process = subprocess.Popen(["mpg123", "-q", "-"], stdin=subprocess.PIPE)

    # Here we pass output_format to stream()
    async for chunk in tts.stream(output_format="audio-24khz-48kbitrate-mono-mp3"):
        if chunk["type"] == "audio":
            process.stdin.write(chunk["data"])
            process.stdin.flush()

    process.stdin.close()
    process.wait()

asyncio.run(main())



# import asyncio
# import edge_tts

# async def main():
#     tts = edge_tts.Communicate("Hello Pravin, it's jarvis nice to meet you, what are you doing here i am fine what about you i think you are great,    byeee.", "en-US-AriaNeural")
#     await tts.save("test.mp3")

# asyncio.run(main())