import multiprocessing
import queue
import time
import json
import logging
import concurrent.futures
import ollama

logger = logging.getLogger(__name__)

# Constants
LLM_MODEL = "qwen2.5:1.5b"
CONTEXT_LIMIT = 2

TRANSLATION_PROMPT_TEMPLATE = """You are a strict JSON translation API. 
Your ONLY task is to translate the following {source_lang} text to {target_lang}.
Output ONLY valid JSON in this exact format: {{"translation": "the translated string"}}.
Do not add any conversational text, greetings, or explanations.

CRITICAL INSTRUCTION 1: Ensure the translation is perfectly punctuated with commas and periods to make it easy to read aloud smoothly. Do not output run-on sentences.
CRITICAL INSTRUCTION 2: If you encounter specialized medical or legal terminology that you do not understand, translate it literally. DO NOT invent or hallucinate medical terms.
CRITICAL INSTRUCTION 3: Translate colloquialisms, idioms, and slang into their natural cultural equivalent meaning, not literally (e.g. Spanish 'me cago en la leche' means 'damn it', 'chanta' means 'scammer').

Use this conversation history to understand context, tone, and idioms:
{context_str}

Text to translate:
{text}
"""

def translate_ollama(text: str, source_lang: str, target_lang: str, context_history: list) -> tuple:
    """Translates text using the Ollama local LLM, forced into JSON mode."""
    start_t = time.time()
    
    # Take only the last CONTEXT_LIMIT phrases to reduce prompt latency
    recent_context = context_history[-CONTEXT_LIMIT:] if context_history else []
    context_str = "\n".join([f"- {h}" for h in recent_context])
    
    prompt = TRANSLATION_PROMPT_TEMPLATE.format(
        source_lang=source_lang,
        target_lang=target_lang,
        context_str=context_str,
        text=text
    )
    try:
        response = ollama.chat(
            model=LLM_MODEL, 
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={'temperature': 0.0}
        )
        # Parse JSON output
        res_json = json.loads(response['message']['content'])
        translated_text = res_json.get('translation', '')
        
        # Fallback if the model failed to output the expected key
        if not translated_text:
            translated_text = response['message']['content']
            
    except Exception as e:
        logger.error(f"Ollama translation failed: {e}")
        translated_text = f"[LLM Error: {e}]"
        
    return translated_text, round(time.time() - start_t, 2)


def process_translation_task(task: tuple, context_history: list, ui_queue: multiprocessing.Queue, timing: dict):
    """Processes a single translation task using Ollama. Propagates pipeline timing."""
    text, lang = task[:2]
    
    target_lang = "Spanish" if lang == "en" else "English"
    source_lang = "English" if lang == "en" else "Spanish"

    timing["translation_start"] = time.time()

    # Run Ollama translation
    ollama_translation, ollama_time = translate_ollama(text, source_lang, target_lang, context_history)
    
    timing["translation_end"] = time.time()
    translation_elapsed = timing["translation_end"] - timing["translation_start"]
    logger.info(f"[TRANSLATOR] Translation completed in {translation_elapsed:.3f}s")

    ui_queue.put({
        "type": "translation",
        "original": text,
        "translated": ollama_translation,
        "latency": ollama_time,
        "timing": timing
    })


def start_translator(translation_queue: multiprocessing.Queue, ui_queue: multiprocessing.Queue):
    """Main process loop for translation."""
    
    # Pre-warm Ollama to load the model into memory
    try:
        logger.info(f"Warming up Ollama with model {LLM_MODEL}...")
        ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': '{"test":"hi"}'}], format='json', options={'temperature': 0.0}, keep_alive=-1)
    except Exception as e:
        logger.warning(f"Failed to pre-warm Ollama: {e}. Is Ollama running?")

    context_history = []
    ui_queue.put({"type": "status", "process": "translator", "status": "ready"})

    # We use a ThreadPoolExecutor to handle incoming requests concurrently without blocking the queue reader
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        while True:
            try:
                task = translation_queue.get()
                if task is None:
                    break

                # Support both 2-element (legacy) and 3-element (with timing) tuples
                if len(task) == 3:
                    text, lang, timing = task
                else:
                    text, lang = task[:2]
                    timing = {}

                context_history.append(text)
                if len(context_history) > 10:
                    context_history.pop(0)

                # Submit to thread pool
                executor.submit(process_translation_task, (text, lang), context_history.copy(), ui_queue, timing)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in translator loop: {e}")
                ui_queue.put({"type": "error", "message": f"Translation Error: {e}"})
