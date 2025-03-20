import torch
from transformers import BitsAndBytesConfig, pipeline
import whisper
import gradio as gr
import time
import warnings
from gtts import gTTS
from PIL import Image
import numpy as np
import whisper
import re
import datetime
import requests
import gradio as gr
import base64
import os
import locale
locale.getpreferredencoding = lambda: "UTF-8"

device = "cuda" if torch.cuda.is_available() else "cpu"

quant_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_compute_dtype=torch.float16,
)

model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline(
    "image-to-text",
    model=model_id,
    model_kwargs = {"quantization_config":quant_config}
)

model = whisper.load_model("medium", device=device)

## logger file

tstamp= datetime.datetime.now()
tstamp= str(tstamp).replace(" ","_")

logfile = f"log_{tstamp}.txt"

def writehistory(text):
    with open(logfile, "a", encoding-'utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()


def img2txt(input_text, input_image):
    image=Image.open(input_image)
    writehistory(f"Input text: {input_text} - Type: {type(input_text)} - Dir: {dir(input_text)}")

    if type(input_text)==tuple:
        prompt_instructions="""
        Describe the image using as much as detail as possible. 
        You are a helpful AI assistent who is able to answer the questions about the image.
        What is the image all about?
        Now generate the helpful answer.
        """
    else:
        prompt_instructions = """
        Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt:
        """

    writehistory(f"prompt_instruction : {prompt_instructions}")
    prompt = "User: <image>\n" + prompt_instructions + "\nAssistant:"
    output = pipe(image,prompt,generate_kwargs={"max_new_tokens":200})

    if output is not None and len(output[0]['generated_text'])>0:
        match = re.search("\nAssistant:\s*(.*)", output[0]("generated_text"))
        if match:
            reply=match.group(1)
        else:
            reply = "No response generated"
    else:
        reply ='No response generated'

    return reply


def transcribe(audio):
    if audio is None or audio=="":
        return ('','',None)

    audio =whisper.load_audio(audio)
    audio =whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _,prob = model.detect_language(mel)

    options= whisper.DecodingOptions()
    result = whisper.decode(model,mel,options)
    result_text=result.text

    return result_text

def text_to_speech(text,file_path):
    language='en'

    audiooj = gTTS(text=text,
                  lang=language,
                  show=False)

    audioobj.save(file_path)
    return file_path

!ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 10 -q:a 9 -acodec libmp3lame Temp.mp3




# A function to handle audio and image inputs
def process_inputs(audio_path, image_path):
    # Process the audio file (assuming this is handled by a function called 'transcribe')
    speech_to_text_output = transcribe(audio_path)

    # Handle the image input
    if image_path:
        mml_output = img2txt(speech_to_text_output, image_path)
    else:
        mml_output = "No image provided."

    # Assuming 'transcribe' also returns the path to a processed audio file
    processed_audio_path = text_to_speech(chatgpt_output, "Temp3.mp3")  # Replace with actual path if different

    return speech_to_text_output, mml_output, processed_audio_path

# Create the interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="AI Output"),
        gr.Audio("Temp.mp3")
    ],
    title="LLM Powered Voice Assistant for Multimodal Data",
    description="Upload an image and interact via voice input and audio response."
)

# Launch the interface
iface.launch(debug=True)
