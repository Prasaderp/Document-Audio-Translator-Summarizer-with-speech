import fitz  # PyMuPDF
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, VitsModel, AutoTokenizer as TTSTokenizer
import docx
from torch import autocast
from functools import lru_cache
import os
import time
import numpy as np
import whisper
import gradio as gr
from huggingface_hub import login
import torchaudio  # For audio speed adjustment
from flask import Flask, request, jsonify
import threading
import subprocess

# Configuration
NLLB_MODEL_NAME = "facebook/nllb-200-3.3B"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MMS_TTS_MODEL_NAME = "facebook/mms-tts-mar"  # Marathi TTS model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MARATHI_CODE = "mar_Deva"
MAX_LENGTH_NLLB = 128
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Hugging Face Token
HF_TOKEN = "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"
login(HF_TOKEN)

# Initialize NLLB model at startup
print("🔄 Initializing NLLB model...")
start_time = time.time()
nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, src_lang="eng_Latn", token=HF_TOKEN)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME, torch_dtype=torch.float16, token=HF_TOKEN).to(DEVICE).eval()
print(f"✅ NLLB model loaded in {time.time() - start_time:.2f}s")

# Initialize LLaMA model at startup
print("🔄 Initializing LLaMA model for summarization...")
start_time = time.time()
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, token=HF_TOKEN)
llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)
summarizer = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer, return_full_text=False, max_new_tokens=512)
print(f"✅ LLaMA model loaded in {time.time() - start_time:.2f}s")

# Initialize MMS-TTS model at startup
print("🔄 Initializing MMS-TTS model for Marathi...")
start_time = time.time()
tts_tokenizer = TTSTokenizer.from_pretrained(MMS_TTS_MODEL_NAME, token=HF_TOKEN)
tts_model = VitsModel.from_pretrained(MMS_TTS_MODEL_NAME, token=HF_TOKEN).to(DEVICE).eval()
print(f"✅ MMS-TTS model loaded in {time.time() - start_time:.2f}s")

# Cache translations
@lru_cache(maxsize=1000)
def cached_translate(text):
    return translate_to_marathi(text)

# Efficient text extraction
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        extracted_text = " ".join(page.get_text("text") for page in doc)
    return extracted_text.strip()

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    extracted_text = " ".join(para.text for para in doc.paragraphs if para.text.strip())
    return extracted_text.strip()

def transcribe_audio(audio_path):
    print("🔄 Initializing Whisper model for audio...")
    start_time = time.time()
    whisper_model = whisper.load_model("base", device=DEVICE)
    print(f"✅ Whisper model loaded in {time.time() - start_time:.2f}s")
    result = whisper_model.transcribe(audio_path, fp16=(DEVICE == "cuda"))
    del whisper_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return result["text"]

# Dynamic batch size for NLLB
def get_dynamic_batch_size(num_chunks):
    if DEVICE != "cuda":
        return min(4, num_chunks)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = total_memory - torch.cuda.memory_allocated()
    tokens_per_chunk = MAX_LENGTH_NLLB
    bytes_per_chunk = tokens_per_chunk * 4
    max_batch = max(1, min(free_memory // bytes_per_chunk, num_chunks))
    return min(8, max_batch)

# Optimized NLLB batch translation
def translate_to_marathi(texts):
    if isinstance(texts, str):
        texts = [texts]

    def split_text(text, max_len=MAX_LENGTH_NLLB):
        words = text.split()
        chunks = []
        current_chunk = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > max_len:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = len(word)
            else:
                current_chunk.append(word)
                current_len += len(word) + 1
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    all_chunks = [chunk for text in texts for chunk in split_text(text)]
    if not all_chunks:
        return "" if len(texts) == 1 else [""] * len(texts)

    batch_size = get_dynamic_batch_size(len(all_chunks))
    translated_chunks = []

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        inputs = nllb_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH_NLLB).to(DEVICE)

        with torch.no_grad(), autocast(device_type=DEVICE if DEVICE == "cuda" else "cpu"):
            outputs = nllb_model.generate(
                **inputs,
                forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(MARATHI_CODE),
                max_length=MAX_LENGTH_NLLB,
                num_beams=3,
                use_cache=True,
                early_stopping=True
            )
        translated_chunks.extend(nllb_tokenizer.batch_decode(outputs, skip_special_tokens=True))
        del inputs, outputs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    translated_texts = []
    chunk_idx = 0
    for text in texts:
        num_chunks = len(split_text(text))
        translated_texts.append(" ".join(translated_chunks[chunk_idx:chunk_idx + num_chunks]))
        chunk_idx += num_chunks

    return translated_texts[0] if len(texts) == 1 else translated_texts

# Summarization with LLaMA
def summarize_text(text):
    if not text:
        return "No text provided for summarization."
    messages = [
        {"role": "system", "content": "You are an AI that provides concise paragraph summaries."},
        {"role": "user", "content": f"Summarize the following text: {text}"}
    ]
    response = summarizer(messages, do_sample=False)
    summary = response[0].get('generated_text', "No summary generated.").strip()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return summary

# Generate audio from Marathi text with MMS-TTS and speed adjustment
def text_to_speech_marathi(text, output_path="marathi_summary.wav", speed=1.2):
    if not text or text == "No input provided":
        return None

    # Tokenize the Marathi text
    inputs = tts_tokenizer(text, return_tensors="pt").to(DEVICE)

    # Generate waveform
    with torch.no_grad():
        audio = tts_model(**inputs).waveform.squeeze().cpu()

    # Convert to numpy array and adjust speed
    audio_np = audio.numpy()
    sample_rate = tts_model.config.sampling_rate  # Typically 16kHz for MMS-TTS

    # Save original audio
    torchaudio.save(output_path, torch.tensor(audio_np).unsqueeze(0), sample_rate)

    # Adjust speed using torchaudio (resampling)
    waveform, orig_sample_rate = torchaudio.load(output_path)
    new_sample_rate = int(orig_sample_rate * speed)
    sped_up_waveform = torchaudio.transforms.Resample(orig_sample_rate, new_sample_rate)(waveform)

    # Save the sped-up audio
    torchaudio.save(output_path, sped_up_waveform, new_sample_rate)

    return output_path

# Combined processing function
def process_input(audio_path=None, doc_path=None):
    extracted_text = ""
    if audio_path:
        extracted_text = transcribe_audio(audio_path)
    elif doc_path:
        if doc_path.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(doc_path)
        elif doc_path.endswith(".docx"):
            extracted_text = extract_text_from_docx(doc_path)
    else:
        return "No input provided", "No input provided", "No input provided", None

    # Translate full extracted text to Marathi
    marathi_text = cached_translate(extracted_text)

    # Summarize in English
    english_summary = summarize_text(extracted_text)

    # Translate summary to Marathi
    marathi_summary = cached_translate(english_summary)

    # Generate audio for Marathi summary with faster speed
    audio_output = text_to_speech_marathi(marathi_summary, speed=1.2)  # 20% faster

    return marathi_text, english_summary, marathi_summary, audio_output

# Gradio interface function
def gradio_process(audio_file, doc_file):
    audio_path = audio_file if audio_file else None
    doc_path = doc_file if doc_file else None
    if not audio_path and not doc_path:
        return "Please upload an audio or document file.", "", "", None

    start_time = time.time()
    marathi_text, english_summary, marathi_summary, audio_output = process_input(audio_path, doc_path)
    print(f"✅ Processing completed in {time.time() - start_time:.2f}s")
    return marathi_text, english_summary, marathi_summary, audio_output

# Flask app setup
app = Flask(__name__)

# Function to setup ngrok
def setup_ngrok():
    try:
        # Download ngrok
        if not os.path.exists("ngrok"):
            print("Downloading ngrok...")
            os.system("wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz")
            os.system("tar -xvf ngrok-v3-stable-linux-amd64.tgz")
            os.system("rm ngrok-v3-stable-linux-amd64.tgz")
            os.system("chmod +x ngrok")

        # Authenticate ngrok (use your ngrok authtoken, sign up at ngrok.com if needed)
        NGROK_AUTH_TOKEN = "2uldAo8FifutZ1A2mQHb1rFy3nx_4dcXtfop8YDrFedeU4fP8"  # Replace with your token
        subprocess.run(["./ngrok", "authtoken", NGROK_AUTH_TOKEN], check=True)

        # Start ngrok tunnel
        print("Starting ngrok tunnel...")
        ngrok_process = subprocess.Popen(["./ngrok", "http", "5000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Wait for ngrok to start

        # Get public URL
        import requests
        tunnels = requests.get("http://localhost:4040/api/tunnels").json()
        public_url = tunnels["tunnels"][0]["public_url"]
        print(f"Flask public URL: {public_url}")
        return ngrok_process, public_url
    except Exception as e:
        print(f"Failed to setup ngrok: {e}")
        print("Falling back to local Flask server at http://127.0.0.1:5000")
        return None, "http://127.0.0.1:5000"

@app.route('/api/process', methods=['POST'])
def http_process():
    audio_file = request.files.get('audio_file')
    doc_file = request.files.get('doc_file')
    
    # Save uploaded files temporarily
    audio_path = None
    doc_path = None
    if audio_file:
        audio_path = "temp_audio_" + str(time.time()) + "." + audio_file.filename.split('.')[-1]
        audio_file.save(audio_path)
    if doc_file:
        doc_path = "temp_doc_" + str(time.time()) + "." + doc_file.filename.split('.')[-1]
        doc_file.save(doc_path)

    # Process the input
    start_time = time.time()
    marathi_text, english_summary, marathi_summary, audio_output = process_input(audio_path, doc_path)
    print(f"✅ HTTP Processing completed in {time.time() - start_time:.2f}s")

    # Prepare response
    response = {
        'marathi_text': marathi_text,
        'english_summary': english_summary,
        'marathi_summary': marathi_summary,
        'audio_output': audio_output  # File path only
    }

    # Clean up temporary files
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
    if doc_path and os.path.exists(doc_path):
        os.remove(doc_path)

    return jsonify(response)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Audio/Document Extraction, Summarization, and Translation to Marathi with Audio Output")
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
        doc_input = gr.File(label="Upload PDF/DOCX")
    process_button = gr.Button("Process")
    marathi_text_box = gr.Textbox(label="Extracted Marathi Text", lines=5)
    summary_box = gr.Textbox(label="English Summary", lines=3)
    marathi_summary_box = gr.Textbox(label="Marathi Summary", lines=3)
    audio_output = gr.Audio(label="Marathi Summary Audio", type="filepath")

    process_button.click(
        fn=gradio_process,
        inputs=[audio_input, doc_input],
        outputs=[marathi_text_box, summary_box, marathi_summary_box, audio_output],
        api_name="/gradio_process"
    )

# Run Flask and Gradio
if __name__ == "__main__":
    # Install dependencies in Colab
    os.system("pip install flask pyngrok")

    # Setup ngrok
    ngrok_process, flask_url = setup_ngrok()

    # Start Flask in a thread
    def run_flask():
        app.run(host="0.0.0.0", port=5000)

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Launch Gradio
    demo.launch(debug=True, share=True, show_error=True)

    # Cleanup ngrok process on exit (if it exists)
    if ngrok_process:
        ngrok_process.terminate()