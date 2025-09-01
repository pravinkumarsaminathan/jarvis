#!/usr/bin/env python3

from assistance.utils import speak
"""
Jarvis Personal AI Hacker Mentor — Safe Buffer Overflow Demo

WHAT THIS DOES (in a controlled, legal, local-only way):
- Explains what a stack-based buffer overflow is.
- Builds two tiny C programs on the fly:
    1) "ssp_enabled"  : compiled with stack canaries and typical hardening flags (default on modern distros)
    2) "no_ssp"       : compiled with -fno-stack-protector to show what happens without canaries
- Runs them with short and overlong inputs to demonstrate: normal run → stack protector abort → or a segfault.
- Narrates each step in plain English (optional text-to-speech via pyttsx3).

This demo is EDUCATIONAL ONLY. It does not craft malicious payloads; it only shows crashing behavior and how mitigations help.

USAGE (on Kali or any Linux with gcc & Python 3):
  python3 jarvis_hacker_mentor.py           # run interactive mentor
  python3 jarvis_hacker_mentor.py --no-tts  # run without voice

Dependencies:
  - gcc (build-essential)
  - Python 3.x
  - Optional: pyttsx3 for offline TTS (pip install pyttsx3)

Author: You + Jarvis 😊
"""

import os
import shutil
import subprocess
import sys
import tempfile
from textwrap import indent

try:
    import pyttsx3  # Optional; used only if available
except Exception:  # pragma: no cover
    pyttsx3 = None


C_SOURCE = r"""
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Deliberately vulnerable function for DEMO PURPOSES ONLY.
// Copies attacker-controlled input into a fixed-size stack buffer without bounds checks.
void vulnerable_copy(const char *input) {
    char buf[16];                     // 16 bytes on the stack
    // UNSAFE: strcpy does not check length. Using it here ONLY to demonstrate overflow.
    strcpy(buf, input);               // If input > 15 chars (+ NUL), this overflows buf
    printf("[program] Copied: %s\n", buf);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input>\n", argv[0]);
        return 1;
    }
    vulnerable_copy(argv[1]);
    puts("[program] Finished normally.");
    return 0;
}
"""


def say(engine, text, also_print=True):
    if also_print:
        print(text)
    if engine is not None:
        speak(text)
        # engine.say(text)
        # engine.runAndWait()


def run(cmd, cwd=None):
    """Run a command and return (returncode, stdout, stderr)."""
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err


def build_binaries(workdir):
    src = os.path.join(workdir, "vuln.c")
    with open(src, "w") as f:
        f.write(C_SOURCE)

    # Likely-hardened build (stack protector enabled, PIE, FORTIFY). On many distros this is default.
    hardened = os.path.join(workdir, "ssp_enabled")
    hard_flags = [
        "gcc", src, "-o", hardened,
        "-O2", "-Wall", "-Wextra",
        "-D_FORTIFY_SOURCE=2",
        "-fstack-protector-strong",
        "-fpie", "-pie"
    ]

    # Intentionally disable the stack protector (for comparison)
    no_ssp = os.path.join(workdir, "no_ssp")
    weak_flags = [
        "gcc", src, "-o", no_ssp,
        "-O0", "-Wall", "-Wextra",
        "-fno-stack-protector",
        "-no-pie"
    ]

    rc1, out1, err1 = run(hard_flags, cwd=workdir)
    rc2, out2, err2 = run(weak_flags, cwd=workdir)

    return {
        "ssp_enabled": {
            "path": hardened,
            "rc": rc1,
            "stdout": out1,
            "stderr": err1,
        },
        "no_ssp": {
            "path": no_ssp,
            "rc": rc2,
            "stdout": out2,
            "stderr": err2,
        },
    }


def demonstrate(engine, workdir):
    say(engine, "Okay. Let's explore a stack-based buffer overflow in a SAFE way.")
    print()

    # Explain concept
    concept = (
        "Concept: A stack buffer overflow happens when a program writes more data than a fixed-size\n"
        "stack buffer can hold. Extra bytes overwrite nearby stack data (like saved frame pointers\n"
        "or canary values). Modern defenses (stack canaries, ASLR, NX, PIE) usually detect or block\n"
        "exploitation, causing an abort instead of silent corruption.\n"
    )
    say(engine, concept)

    say(engine, "Step 1: Building two tiny C programs — one with stack canaries, one without …")
    builds = build_binaries(workdir)
    if builds["ssp_enabled"]["rc"] != 0 or builds["no_ssp"]["rc"] != 0:
        print("Compiler output (ssp_enabled):", builds["ssp_enabled"]["stderr"].strip())
        print("Compiler output (no_ssp):", builds["no_ssp"]["stderr"].strip())
        raise SystemExit("Failed to build demo programs. Ensure gcc is installed.")

    say(engine, "Built successfully.")
    print()

    # Inputs: safe, overflow16+, and bigger
    tests = [
        ("Short input (<=15 chars)", "HelloWorld"),
        ("Overflow input (~32 chars)", "A" * 32),
        ("Bigger overflow (~128 chars)", "B" * 128),
    ]

    def run_prog(tag, path, arg):
        rc, out, err = run([path, arg], cwd=workdir)
        merged = (out + err).strip()
        print(f"\n--- {tag} | argv length = {len(arg)} ---\n" + indent(merged, "  "))
        return rc, merged

    say(engine, "Step 2: Run the hardened binary with canaries enabled …")
    for label, arg in tests:
        run_prog(f"ssp_enabled → {label}", builds["ssp_enabled"]["path"], arg)

    say(engine, "Observation: With canaries, overlong input is detected — you'll likely see 'stack smashing detected' and an abort.")

    say(engine, "Step 3: Now compare with a build that DISABLES the stack protector …")
    for label, arg in tests:
        run_prog(f"no_ssp → {label}", builds["no_ssp"]["path"], arg)

    say(engine, (
        "Observation: Without canaries, the program may crash with a segmentation fault or show corrupted behavior,\n"
        "illustrating why modern mitigations matter.\n"
    ))

    # Defensive best practices
    best = (
        "Defensive takeaways:\n"
        "  • NEVER use unchecked copies like strcpy/gets; prefer strncpy/strlcpy or, better, safe APIs.\n"
        "  • Validate lengths from untrusted input.\n"
        "  • Compile with hardening: -fstack-protector-strong, -D_FORTIFY_SOURCE=2, PIE, RELRO, and enable ASLR.\n"
        "  • Use memory-safe languages (Rust, Go) where possible for new code.\n"
    )
    say(engine, best)

    say(engine, "Demo complete. This showed overflow-caused crashes — not exploitation.")


def main():
    tts = True
    if "--no-tts" in sys.argv:
        tts = False

    engine = None
    if tts and pyttsx3 is not None:
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 185)
            engine.setProperty('volume', 0.9)
        except Exception:
            engine = None

    # Create an isolated temp workspace
    workdir = tempfile.mkdtemp(prefix="jarvis_bof_demo_")
    try:
        demonstrate(engine, workdir)
        print("\nWorkspace:", workdir)
        print("(You can inspect binaries and source here; it will NOT be auto-deleted so you can review.)")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Keep the directory so the user can inspect artifacts; uncomment below to auto-clean
        # shutil.rmtree(workdir, ignore_errors=True)
        pass


if __name__ == "__main__":
    main()
