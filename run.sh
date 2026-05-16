#!/bin/bash


cd ~/projects/sir/sir-tts-whisper/
source .venv/bin/activate && fastapi run main.py
