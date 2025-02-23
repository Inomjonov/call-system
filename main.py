from fastapi import FastAPI, HTTPException,File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import io
import torch
from transformers import AutoProcessor, AutoModelForTextToSpectrogram, SpeechT5HifiGan, pipeline
import librosa
import numpy as np
import soundfile as sf
import asyncio
from openai import OpenAI
app = FastAPI(title="call-centre")

#here starts tts
# model_dir = "model"
# processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
# model = AutoModelForTextToSpectrogram.from_pretrained(model_dir, local_files_only=True)

# # Load vocoder
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# @app.get("/text-to-speech/")
# async def synthesize_speech(text: str):
#     try:
#         # Load speaker embeddings
#         speaker_embeddings = torch.load(f"{model_dir}/speaker_embeddings.pt")
#         # Process text and generate speech
#         inputs = processor(text=text, return_tensors="pt")
#         speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
#         # Convert tensor to numpy array
#         speech_array = speech.numpy()
        
#         if not np.isfinite(speech_array).all():
#             raise ValueError("Generated audio contains NaN or infinite values.")        
        
#         # Original parameters
#         original_rate = 16000
#         speed_factor = 0.7

#         speech_array = speech_array.astype(np.float32)
       
#         if speed_factor != 1.0:
#             speech_array = librosa.effects.time_stretch(speech_array, rate=1 / speed_factor)
        
#         audio_buffer = io.BytesIO()
#         sf.write(audio_buffer, speech_array, samplerate=int(original_rate * speed_factor), format='WAV')
#         audio_buffer.seek(0)

#         return StreamingResponse(audio_buffer, media_type="audio/wav")

#     except ValueError as e:
#         return HTTPException(status_code=500, detail=f"Invalid audio data: {str(e)}")

#     except Exception as e:
#         return HTTPException(status_code=500, detail=f"Error processing speech: {str(e)}")



# here starts stt
auth_token = "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"
stt_model = pipeline("automatic-speech-recognition", model="Inomjonov/mironshoh-whisper-medium-uz-10k", token=auth_token)

async def text_streamer(contents):
    """
    Generator function that processes the audio and yields text in chunks.
    """
    audio_buffer = io.BytesIO(contents)
    data, rate = sf.read(audio_buffer, dtype="float32")
    inputs = {"raw": data, "sampling_rate": rate}

    # Simulate streaming by processing the complete audio and yielding text in chunks
    result = stt_model(inputs)
    text = result['text']
    chunk_size = 50  # Send chunks of 50 characters
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size] + "\n"
        await asyncio.sleep(0.1)  # Optional: simulate delay

@app.post("/speech-to-text/")
async def speech_to_text(file: UploadFile = File(...)):
    # Read audio file
    contents = await file.read()

    # Create a StreamingResponse object that streams the text
    return StreamingResponse(content=text_streamer(contents), media_type="text/plain")

# here starts openai call
OPENAI_API_KEY = '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'
client = OpenAI(api_key=OPENAI_API_KEY)

@app.post("/openai-call")
def chat_openai(prompt:str):
    completion = client.chat.completions.create(
        model = "gpt-4o",
        messages=[
            {"role":"developer", "content":"You are call asistant. Answer to questions in Uzbek"},
            {"role":"user", "content":prompt}
        ]
    )
    return completion.choices[0].message

