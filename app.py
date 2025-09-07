import gradio as gr
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from pydub import AudioSegment
from pydub.silence import detect_silence
import yt_dlp
import requests
import validators
from urllib.parse import urlparse
import subprocess
import os
import re
import glob
import spaces
from pathlib import Path

### Initializations

MAX_TOKENS = 32000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"*** Device: {device}")
model_name = 'mistralai/Voxtral-Mini-3B-2507'

processor = AutoProcessor.from_pretrained(model_name)
model = VoxtralForConditionalGeneration.from_pretrained(model_name,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map=device)
# Supported languages
dict_languages = {"English": "en",
                  "French": "fr",
                  "German": "de",
                  "Spanish": "es",
                  "Italian": "it",
                  "Portuguese": "pt",
                  "Dutch": "nl",
                  "Hindi": "hi"}

# Whitelist of allowed MIME types for audio and video
ALLOWED_MIME_TYPES = {
    # Audio
    'audio/mpeg', 'audio/wav', 'audio/wave', 'audio/x-wav', 'audio/x-pn-wav',
    'audio/ogg', 'audio/vorbis', 'audio/aac', 'audio/mp4', 'audio/flac',
    'audio/x-flac', 'audio/opus', 'audio/webm',
    # Video
    'video/mp4', 'video/mpeg', 'video/ogg', 'video/webm', 'video/quicktime',
    'video/x-msvideo', 'video/x-matroska'
}

# Maximum allowed file size (in bytes). Ex: 1 GB
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB

# Directory where the files will be saved
DOWNLOAD_DIR = "downloaded_files"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

MAX_LEN = 1800000 # 30 mn
one_second_silence = AudioSegment.silent(duration=1000)

#### Functions

@spaces.GPU
def chunks_creation(audio_path):
    list_audio_path = [audio_path]
    audio = AudioSegment.from_file(audio_path)
    status = gr.Markdown("👍 Audio duration less than max")
    # Input too large ?
    if len(audio) > MAX_LEN:
        list_audio_path = []
        try:
            # Create list of chunks
            list_silent = detect_silence(audio,min_silence_len=300,
                    # silent if quieter than -14 dBFS threshold
                    silence_thresh=audio.dBFS-14, seek_step=100)
            list_interval = [(start, stop) for start, stop in list_silent]

            # Calculate speech intervals
            list_speech = []
            current_start = 0
            for start, stop in list_interval:
                if current_start < start:
                    list_interval.append((current_start, start))
                current_start = stop
            # Add last interval if needed
            if current_start < len(audio):
                list_speech.append((current_start, len(audio)))

            # Determination of chunks, to fit within the maximum duration
            list_chunks = []
            deb_chunk, fin_chunk = 0, list_speech[0][1]

            for start, end in list_speech[1:]:
                if end - deb_chunk + one_second_silence <= MAX_LEN:
                    fin_chunk = end + one_second_silence
                else:
                    list_chunks.append([deb_chunk, fin_chunk])
                    deb_chunk, fin_chunk = start, end
            list_chunks.append([deb_chunk, fin_chunk+one_second_silence])

            # Save chunks
            for i, (start, stop) in enumerate(list_chunks):
                segment = audio[start:stop]
                segment.export(f"chunk_{i}.wav", format="wav")
                list_audio_path.append(f"chunk_{i}.wav")

            status = f"✅ **Success!** {len(list_audio_path)} chunks saved."
        except Exception as e:
            status = gr.Markdown(f"❌ **Unexpected error during chuncks creation:** {e}")

    return list_audio_path, status
###

@spaces.GPU
def process_transcript(language: str, audio_path: str) -> str:
    """Process the audio file to return its transcription.

    Args:
        language: The language of the audio.
        audio_path: The path to the audio file.

    Returns:
        The transcribed text of the audio.
        The status of transcription : with or without chunking.
    """
    result = ""
    status = gr.Markdown()

    if audio_path is None:
        status = gr.Markdown("Please provide some input audio: either upload an audio file or use the microphone.")
    else:
        id_language = dict_languages[language]

        # Verification of the duration, for possible division into chunks
        list_audio_path, status = chunks_creation(audio_path)

        # Transcription process
        try:
            for path in list_audio_path:
                inputs = processor.apply_transcription_request(language=id_language,
                                                              audio=path, model_id=model_name)
                inputs = inputs.to(device, dtype=torch.bfloat16)
                outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
                decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                                         skip_special_tokens=True)
                result += decoded_outputs[0]
            status = "✅ **Success!** Transcription done."
        except Exception as e:
            status = gr.Markdown(f"❌ **Unexpected error during transcription:** {e}")

    return result, status
###

@spaces.GPU
def process_translate(language: str, audio_path: str) -> str:
    result = ""
    status = gr.Markdown()

    if audio_path is None:
        status = gr.Markdown("Please provide some input audio: either upload an audio file or use the microphone.")
    else:
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": audio_path,
                        },
                        {"type": "text", "text": "Translate this in "+language},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(conversation)
            inputs = inputs.to(device, dtype=torch.bfloat16)

            outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
            decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            result = decoded_outputs[0]
            status = "✅ **Success!** Translation done."
        except Exception as e:
            status = gr.Markdown(f"❌ **Unexpected error during translation:** {e}")

    return result, status
###

@spaces.GPU
def process_chat(question: str, audio_path: str) -> str:
    result = ""
    status = gr.Markdown()

    if audio_path is None:
        status = gr.Markdown("Please provide some input audio: either upload an audio file or use the microphone.")
    else:
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": audio_path,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(conversation)
            inputs = inputs.to(device, dtype=torch.bfloat16)

            outputs = model.generate(**inputs, max_new_tokens=500)
            decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

            result = decoded_outputs[0]
            status = "✅ **Success!**"
        except Exception as e:
            status = gr.Markdown(f"❌ **Unexpected error during chat process:** {e}")

    return result, status
###

def disable_buttons():
    return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

def enable_buttons():
    return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
###

def clear_audio():
    return None, None, None, None
###

@spaces.GPU
def voice_extract_demucs(audio_path):
    """
    Returns the path of the voice extracted file.
    """
    try:
        os.makedirs("out_demucs", exist_ok=True)

        cmd = [
            "demucs",
            "--two-stems=vocals",
            "--out", "out_demucs",
            audio_path
        ]
        subprocess.run(cmd, check=True)
        base_name = Path(audio_path).stem
        model_dirs = list(Path("out_demucs").glob("*"))
        if not model_dirs:
            vocals_path = audio_path
            success_message = "❌ **Error:** No files found on out_demucs."
        else:
            result_dir = model_dirs[0] / base_name
            vocals_path = result_dir / "vocals.wav"
            success_message = "✅ **Success!** Voice extracted."

        return str(vocals_path), str(vocals_path), gr.Markdown(success_message)
    except Exception as e:
        return None, None, gr.Markdown(f"❌ **Error:** An unexpected ERROR occurred: {e}")
###

def secure_download_from_url(url: str):
    """
    Validates a URL and downloads the file if it is an authorized media.
    Returns the path of the downloaded file or an error message.
    """
    # Step 1: Validate the URL format
    if not validators.url(url):
        return None, None, gr.Markdown("❌ **Error:** The provided URL is invalid.")

    try:
        # Step 2: Send a HEAD request to check the headers without downloading the content
        # allow_redirects=True to follow redirects to the final file location.
        # timeout to avoid blocking requests.
        response = requests.head(url, allow_redirects=True, timeout=10)

        # Check if the request was successful (status code 2xx)
        response.raise_for_status()

        # Step 3: Validate the content type (MIME type)
        content_type = response.headers.get('Content-Type', '').split(';')[0].strip()
        if content_type not in ALLOWED_MIME_TYPES:
            error_message = (
                 f"❌ **Error:** The file type is not allowed.\n"
                 f" - **Type detected:** `{content_type}`\n"
                 f" - **Allowed types:** Audio and Video only."
            )
            return None, None, gr.Markdown(error_message)

        # Step 4: Validate the file size
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > MAX_FILE_SIZE:
            error_message = (
                f"❌ **Error:** The file is too large.\n"
                f" - **File size:** {int(content_length) / 1024 / 1024:.2f} MB\n"
                f" - **Maximum allowed size:** {MAX_FILE_SIZE / 1024 / 1024:.2f} MB"
            )
            return None, None, gr.Markdown(error_message)

        # Step 5: Secure streaming download
        with requests.get(url, stream=True, timeout=20) as r:
            r.raise_for_status()

            # Extract the file name from the URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename: # Si l'URL se termine par un '/'
                filename = "downloaded_media_file"

            filepath = os.path.join(DOWNLOAD_DIR, filename)

            # --- Step 6: Download the audio ---
            # Write the file in chunks to avoid overloading memory
            with open(filepath, 'wb') as f:
                downloaded_size = 0
                for chunk in r.iter_content(chunk_size=8192):
                    downloaded_size += len(chunk)
                    if downloaded_size > MAX_FILE_SIZE:
                         os.remove(filepath) # Supprimer le fichier partiel
                         return None, None, gr.Markdown("❌ **Error:** The file exceeds the maximum allowed size during download.")
                    f.write(chunk)

        # --- Step 7: Convert to WAV using Pydub ---
        audio_file = AudioSegment.from_file(filepath)
        file_handle = audio_file.export("audio_file.wav", format="wav")

        # --- Step 8: Clean up ---
        try:
            files = glob.glob(DOWNLOAD_DIR)
            for f in files:
                os.remove(f)
        except:
            pass

        success_message = (
            f"✅ **Success!** File downloaded and saved."
        )

        # Returns the file path and a success message.
        return "audio_file.wav", "audio_file.wav", gr.Markdown(success_message)

    except requests.exceptions.RequestException as e:
        # Handle network errors (timeout, DNS, connection refused, etc.)
        return None, None, gr.Markdown(f"❌ **Network error:** Unable to reach URL. Details: {e}")
    except Exception as e:
        # Handle Other potential errors
        return None, None, gr.Markdown(f"❌ **Unexpected error:** {e}")
###

def secure_download_youtube_audio(url: str):
    """
    Returns the path of the downloaded file or an error message.
    """
    # --- Step 1: Validate URL format with Regex ---
    youtube_regex = re.compile(
        r'^(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    if not youtube_regex.match(url):
        return None, None, gr.Markdown("❌ **Error:** The URL '{url}' does not appear to be a valid YouTube URL.")

    try:
        # --- Step 2: Check video availability ---
        ydl_info_opts = {'quiet': True, 'skip_download': True}
        try:
            with yt_dlp.YoutubeDL(ydl_info_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            return None, None, gr.Markdown(f"❌ **Error:** The video at URL '{url}' is unavailable ({str(e)})")

        # --- Step 3: Select best audio format ---
        formats = [f for f in info['formats'] if f.get('acodec') != 'none']
        if not formats:
            return None, None, gr.Markdown("❌ **Error:** No audio-only stream was found for this video.")

        formats.sort(key=lambda f: f.get('abr') or 0, reverse=True)
        best_audio_format = formats[0]

        # --- Step 4: Check file size BEFORE downloading ---
        filesize = best_audio_format.get('filesize') or best_audio_format.get('filesize_approx')
        if filesize is None:
            print("Could not determine file size before downloading.")
            filesize = 1

        if filesize > MAX_FILE_SIZE:
            return None, None, gr.Markdown(
                f"❌ **Error:** The file is too large.\n"
                f" - **File size:** {filesize / 1024 / 1024:.2f} MB\n"
                f" - **Maximum allowed size:** {MAX_FILE_SIZE / 1024 / 1024:.2f} MB"
            )

        # --- Step 5: Download & convert directly to WAV ---
        ydl_opts = {
            'quiet': True,
            'format': f"{best_audio_format['format_id']}",
            'outtmpl': "audio_file",  # will be replaced by ffmpeg output
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        success_message = "✅ **Success!** Audio extracted and saved."
        return "audio_file.wav", "audio_file.wav", gr.Markdown(success_message)

    except FileNotFoundError:
        return None, None, gr.Markdown("❌ **Error:** FFmpeg not found. Please ensure it is installed and in your system's PATH.")
    except Exception as e:
        return None, None, gr.Markdown(f"❌ **Error:** An unexpected ERROR occurred: {e}")
###

def get_sel_audio(audio_path: str) -> str:
    return audio_path, gr.Markdown("✅ **Original** audio is considered.")
###

#### Gradio interface
with gr.Blocks(title="Voxtral") as voxtral:
    with gr.Row():
        gr.Markdown("# **Voxtral Mini Evaluation**")

        with gr.Accordion("🔎 More on Voxtral", open=False):
            gr.Markdown("""## **Key Features:**

#### Voxtral builds upon Ministral-3B with powerful audio understanding capabilities.
##### - **Dedicated transcription mode**: Voxtral can operate in a pure speech transcription mode to maximize performance. By default, Voxtral automatically predicts the source audio language and transcribes the text accordingly
##### - **Long-form context**: With a 32k token context length, Voxtral handles audios up to 30 minutes for transcription, or 40 minutes for understanding
##### - **Built-in Q&A and summarization**: Supports asking questions directly through audio. Analyze audio and generate structured summaries without the need for separate ASR and language models
##### - **Natively multilingual**: Automatic language detection and state-of-the-art performance in the world’s most widely used languages (English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian)
##### - **Function-calling straight from voice**: Enables direct triggering of backend functions, workflows, or API calls based on spoken user intents
##### - **Highly capable at text**: Retains the text understanding capabilities of its language model backbone, Ministral-3B""")

    gr.Markdown("""#### Voxtral Mini is an enhancement of **Ministral 3B**, incorporating state-of-the-art audio input \
    capabilities while retaining best-in-class text performance. It excels at speech transcription, translation and \
    audio understanding. Available languages: English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian.""")

    gr.Markdown("### **1.Choose the audio:**")
    sel_audio = gr.State()
    with gr.Row():
        with gr.Tabs():
            with gr.Tab("From record or file upload"):
                gr.Markdown("### **Upload an audio file, record via microphone, or select a demo file:**")
                gr.Markdown("### *(Voxtral handles audios up to 30 minutes for transcription; if longer, it will be cut into chunks)*")
                sel_audio1 = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                    label="Set an audio file to process it:")
                example1 = [["mapo_tofu.mp3"]]
                gr.Examples(
                    examples=example1,
                    inputs=sel_audio1,
                    outputs=None,
                    fn=None,
                    cache_examples=False,
                    run_on_click=False
                )
                status_output1 = gr.Markdown()
                with gr.Row():
                    voice_button0 = gr.Button("Process original audio", variant="primary")
                    voice_button0.click(
                        fn=get_sel_audio,
                        inputs=sel_audio1,
                        outputs=[sel_audio, status_output1])
                    voice_button1 = gr.Button("Extract voice (if noisy environment)")
                    voice_button1.click(
                        fn=voice_extract_demucs,
                        inputs=sel_audio1,
                        outputs=[sel_audio, sel_audio1, status_output1])
                    clear_audio1 = gr.Button("Clear audio")
                    clear_audio1.click(
                        fn=clear_audio,
                        outputs=[sel_audio, sel_audio, sel_audio1, status_output1])

            with gr.Tab("From file url (audio or video file)"):
                gr.Markdown("### **Enter the url of the file (mp3, wav, mp4, ...):**")
                url_input2 = gr.Textbox(label="URL (MP3 or MP4 file)",
                                       placeholder="https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/mapo_tofu.mp4")
                example2 = [["https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/mapo_tofu.mp4"]]
                gr.Examples(
                    examples=example2,
                    inputs=url_input2,
                    outputs=None,
                    fn=None,
                    cache_examples=False,
                    run_on_click=False
                )
                download_button2 = gr.Button("Check and upload", variant="primary")
                input_audio2 = gr.Audio()
                status_output2 = gr.Markdown()
                download_button2.click(
                    fn=secure_download_from_url,
                    inputs=url_input2,
                    outputs=[input_audio2, sel_audio, status_output2]
                )
                with gr.Row():
                    voice_button2 = gr.Button("Extract voice (if noisy environment)")
                    voice_button2.click(
                        fn=voice_extract_demucs,
                        inputs=input_audio2,
                        outputs=[input_audio2, sel_audio, status_output2])
                    clear_audio1 = gr.Button("Clear audio")
                    clear_audio1.click(
                        fn=clear_audio,
                        outputs=[sel_audio, sel_audio, input_audio2, status_output2])

            with gr.Tab("From Youtube url:"):
                gr.Markdown("### **Enter the url of the Youtube video:**")
                url_input3 = gr.Textbox(label="Youtube url",
                                       placeholder="https://www.youtube.com/...")
                download_button3 = gr.Button("Check and upload", variant="primary")
                input_audio3 = gr.Audio()
                status_output3 = gr.Markdown()
                download_button3.click(
                    fn=secure_download_youtube_audio,
                    inputs=url_input3,
                    outputs=[input_audio3, sel_audio, status_output3]
                )
                with gr.Row():
                    voice_button3 = gr.Button("Extract voice (if noisy environment)")
                    voice_button3.click(
                        fn=voice_extract_demucs,
                        inputs=input_audio3,
                        outputs=[input_audio3, sel_audio, status_output3])
                    clear_audio1 = gr.Button("Clear audio")
                    clear_audio1.click(
                        fn=clear_audio,
                        outputs=[sel_audio, sel_audio, input_audio3, status_output3])

    with gr.Row():
        gr.Markdown("### **2. Choose one of theese tasks:**")

    with gr.Row():
        with gr.Column():
            with gr.Accordion("📝 Transcription", open=True):
                sel_language = gr.Dropdown(
                    choices=list(dict_languages.keys()),
                    value="English",
                    label="Select the language of the audio file:"
                )
                submit_transcript = gr.Button("Extract transcription", variant="primary")
                text_transcript = gr.Textbox(label="💬 Generated transcription", lines=10)
                status_transcript = gr.Markdown()

        with gr.Column():
            with gr.Accordion("🔁 Translation", open=True):
                list_language = list(dict_languages.keys())
                list_language.pop(list_language.index(sel_language.value)) # Fix: Access the value of the dropdown
                sel_translate_language = gr.Dropdown(
                    choices=list(dict_languages.keys()),
                    value="English",
                    label="Select the language for translation:"
                )
                submit_translate = gr.Button("Translate audio file", variant="primary")
                text_translate = gr.Textbox(label="💬 Generated translation", lines=10)
                status_translate = gr.Markdown()

        with gr.Column():
            with gr.Accordion("🤖 Ask audio file", open=True):
                question_chat = gr.Textbox(label="Enter your question about audio file:", placeholder="Enter your question about audio file")
                submit_chat = gr.Button("Ask audio file", variant="primary")
                example_chat = [["What is the subject of this audio file?"], ["Quels sont les ingrédients ?"]]
                gr.Examples(
                    examples=example_chat,
                    inputs=question_chat,
                    outputs=None,
                    fn=None,
                    cache_examples=False,
                    run_on_click=False
                )
                text_chat = gr.Textbox(label="💬 Model answer", lines=10)
                status_chat = gr.Markdown()

### Processing

    # Transcription
    submit_transcript.click(
        disable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
        trigger_mode="once",
    ).then(
        fn=process_transcript,
        inputs=[sel_language, sel_audio],
        outputs=[text_transcript, status_transcript]
    ).then(
        enable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
    )

    # Translation
    submit_translate.click(
        disable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
        trigger_mode="once",
    ).then(
        fn=process_translate,
        inputs=[sel_translate_language, sel_audio],
        outputs=[text_translate, status_translate]
    ).then(
        enable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
    )

    # Chat
    submit_chat.click(
        disable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
        trigger_mode="once",
    ).then(
        fn=process_chat,
        inputs=[question_chat, sel_audio],
        outputs=[text_chat, status_chat]
    ).then(
        enable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
    )

### Launch the app

if __name__ == "__main__":
    voxtral.queue().launch(debug=True)
