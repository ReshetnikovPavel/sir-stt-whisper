import os
from typing import Annotated, BinaryIO

from fastapi import FastAPI, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel
from pydantic import BaseModel


class InputTokenDetails(BaseModel):
    text_tokens: int
    audio_tokens: int


class Usage(BaseModel):
    type: str
    input_tokens: int
    input_token_details: InputTokenDetails
    output_tokens: int
    total_tokens: int


class CreateTranscriptionResponse(BaseModel):
    text: str
    # usage: Usage


root = "models"
model_name = "whisper-tiny-ru-ct2"
path = os.path.join(root, model_name)
model = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=8)


def list_model_names():
    return [model_name]


def transcribe(file: BinaryIO) -> str:
    segments, info = model.transcribe(file)
    return "".join([s.text for s in segments])


app = FastAPI()


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile, model: Annotated[str, Form()]
) -> CreateTranscriptionResponse:  # noqa: F821
    if model not in list_model_names():
        raise HTTPException(404, "Unexcpected model name")
    text = transcribe(file.file)
    return CreateTranscriptionResponse(text=text)
