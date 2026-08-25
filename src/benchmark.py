import time
import json
import ollama
import torch
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Test data
TEST_PHRASES = [
    {"lang": "en", "text": "Hello, how are you today? I hope you are doing well."},
    {"lang": "es", "text": "Hola, ¿cómo estás? Quería decirte que me caes muy mal, te odio, eres un estúpido, un pendejo y un bruto."},
    {"lang": "es", "text": "Le cuento doctor que el día de ayer me caí y me volví la cabeza muy fuerte. Desde ese momento he sentido una perpetación en este lado del cráneo, en toda la parte frontal, como puede ver."}
]

OLLAMA_MODELS = ["llama3.2:3b", "qwen2.5:1.5b", "qwen2.5:0.5b"]

def benchmark_marian():
    logger.info("\n--- Benchmarking MarianMT ---")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading MarianMT on {device}...")
    
    start_load = time.time()
    en_to_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=device)
    es_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=device)
    logger.info(f"Load time: {time.time() - start_load:.2f}s")

    # Warmup
    en_to_es("Warm up")
    
    for idx, item in enumerate(TEST_PHRASES):
        lang = item["lang"]
        text = item["text"]
        start_t = time.time()
        
        if lang == "en":
            res = en_to_es(text)[0]['translation_text']
        else:
            res = es_to_en(text)[0]['translation_text']
            
        latency = time.time() - start_t
        logger.info(f"[{latency:.2f}s] MarianMT: {res}")


def benchmark_ollama(model_name):
    logger.info(f"\n--- Benchmarking Ollama ({model_name}) ---")
    
    # Warmup
    try:
        ollama.chat(model=model_name, messages=[{'role': 'user', 'content': '{"test":"hi"}'}], format='json', options={'temperature': 0.0})
    except Exception as e:
        logger.error(f"Failed to load {model_name}. Ensure it is pulled. ({e})")
        return

    for idx, item in enumerate(TEST_PHRASES):
        lang = item["lang"]
        text = item["text"]
        
        target_lang = "Spanish" if lang == "en" else "English"
        source_lang = "English" if lang == "en" else "Spanish"

        prompt = f"""You are a strict JSON translation API. 
Your ONLY task is to translate the following {source_lang} text to {target_lang}.
Output ONLY valid JSON in this exact format: {{"translation": "the translated string"}}.
Do not add any conversational text, greetings, or explanations.

CRITICAL INSTRUCTION: Ensure the translation is perfectly punctuated with commas and periods to make it easy to read aloud smoothly. Do not output run-on sentences.

Text to translate:
{text}
"""
        start_t = time.time()
        try:
            response = ollama.chat(
                model=model_name, 
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.0}
            )
            res_json = json.loads(response['message']['content'])
            res = res_json.get('translation', 'Error')
        except Exception as e:
            res = f"Error: {e}"

        latency = time.time() - start_t
        logger.info(f"[{latency:.2f}s] {model_name}: {res}")

if __name__ == "__main__":
    logger.info("Starting Benchmark Suite...")
    benchmark_marian()
    for model in OLLAMA_MODELS:
        benchmark_ollama(model)
    logger.info("\nBenchmark Complete.")
