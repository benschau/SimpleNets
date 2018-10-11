# -*- coding: utf-8 -*-

"""
This module is intended to be a simple, naive way of setting up voice controls for my PC.
Meant to be a demo in turning my PC off or to sleep without having to press anything.

Example Usage:
    python PControl.py

TODO:
    * Hoping to eventually train a neural network to listen to just my voice.

"""

import subprocess
import time

import speech_recognition as sr

DEFAULT_DEVICE_INDEX = 7

def transcribe_speech(recognizer, mic):
    """
    Args:
        recognizer - a speech_recognition object, an interface for speech recognition API calls
        mic - a speech_recognition object representing the microphone

    Returns:
        a dictionary with three keys:
            "success":    a True or False, depending on whether the operation succeeded.
            "error":      a string with the contents of the error message, and None otherwise.
                          either the API call failed or no transcription was possible.
            "transcript": a transcription of the recorded audio, if the speech recognition succeeded;
                          None otherwise.
    """

    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be a `Recognizer` instance")

    if not isinstance(mic, sr.Microphone):
        raise TypeError("`mic` must be a `Microphone` instance")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    response = {
            "success": True,
            "error": None,
            "transcript": None
    }

    try:
        response["transcript"] = recognizer.recognize_google(audio) # requires an internet connection
    except sr.RequestError:
        response["success"] = False
        response["error"] = "Couldn't get access to the API."
    except sr.UnknownValueError:
        response["success"] = False
        response["error"] = "Couldn't recognize speech."

    return response

def sleep():
    """
        Put the computer to sleep.
    """
    print("sleep")
    # subprocess.run([""])

def shutdown():
    """
        Shutdown the computer.
    """
    print("shutdown")

if __name__ == "__main__":
    INVOKER = "computer"
    INSTRUCTIONS = {"sleep": sleep, "shut down": shutdown}

    recog = sr.Recognizer()
    mic = sr.Microphone(device_index=DEFAULT_DEVICE_INDEX)

    while True:
        # wait for the invoker:
        invoked = True
        while invoked:
            print("Waiting for invoker, {}...".format(INVOKER))
            result = transcribe_speech(recog, mic)

            if not result["success"] and result["error"]:
                continue

            if result["transcript"]:
                instruct = result["transcript"].lower()

                if instruct == INVOKER:
                    invoked = False


        # wait for the instruction:
        print("Waiting for instruction, sleep or shutdown")
        result = transcribe_speech(recog, mic)

        if not result["success"] and result["error"]:
            continue

        if result["transcript"]:
            instruct = result["transcript"].lower()

            if instruct in INSTRUCTIONS.keys():
                INSTRUCTIONS[instruct]()

