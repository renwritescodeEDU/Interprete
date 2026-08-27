import multiprocessing
import queue
import time
import json
import logging
import concurrent.futures
import ollama

from src.glossary import get_glossary_manager

logger = logging.getLogger(__name__)

# Constants
LLM_MODEL = "qwen2.5:3b"
CONTEXT_LIMIT = 5

TRANSLATION_PROMPT_TEMPLATE = """You are a professional simultaneous interpreter API for live customer service calls.
Translate the {source_lang} text below to {target_lang}. Output ONLY valid JSON: {{"translation": "..."}}

MANDATORY RULES:
1. Use FORMAL register: "usted/su/le" in Spanish, NEVER "tú/tu/te" unless the original speaker uses informal.
2. Translate ALL compound terms naturally: "mother-in-law"="suegra", "checking account"="cuenta corriente", "father-in-law"="suegro", "daughter-in-law"="nuera", "brother-in-law"="cuñado".
3. For acronyms: write the Spanish acronym with full meaning in parentheses on first use. Example: "CD" → "CD (Certificado de Depósito)", "APR" → "TAP (Tasa Anual de Porcentaje)".
4. NEVER leave English words untranslated except proper nouns and brand names.
5. Respect grammatical gender: "un diagnóstico" (masc), "una receta" (fem), "el saldo" (masc).
6. Preserve exact numbers, dates, account numbers, and alphanumeric codes unchanged.
7. Punctuate perfectly with commas and periods for smooth read-aloud delivery.
8. "anyone else besides" = "alguien más aparte de" (NOT "nadie más que").
9. "debited" = "debitado/descontado" (NOT "debilitado").
10. "I need to know if" = "Necesito saber si" (followed by positive construction, NOT double negative).

{glossary_section}

Conversation context:
{context_str}

Translate this text:
{text}
"""


def translate_ollama(text: str, source_lang: str, target_lang: str,
                     context_history: list, glossary_manager=None) -> tuple:
    """Translates text using the Ollama local LLM with glossary-enhanced prompts."""
    start_t = time.time()

    # Take only the last CONTEXT_LIMIT phrases to reduce prompt latency
    recent_context = context_history[-CONTEXT_LIMIT:] if context_history else []
    context_str = "\n".join([f"- {h}" for h in recent_context])

    # Build glossary section from the glossary manager
    glossary_section = ""
    if glossary_manager:
        try:
            glossary_section = glossary_manager.build_glossary_prompt_section(
                text, recent_context
            )
        except Exception as e:
            logger.warning(f"Glossary lookup failed: {e}")
            glossary_section = ""

    prompt = TRANSLATION_PROMPT_TEMPLATE.format(
        source_lang=source_lang,
        target_lang=target_lang,
        glossary_section=glossary_section,
        context_str=context_str if context_str else "(No prior context)",
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


def process_translation_task(task: tuple, context_history: list,
                             ui_queue: multiprocessing.Queue, timing: dict,
                             glossary_manager=None):
    """Processes a single translation task using Ollama. Propagates pipeline timing."""
    text, lang = task[:2]

    target_lang = "Spanish" if lang == "en" else "English"
    source_lang = "English" if lang == "en" else "Spanish"

    timing["translation_start"] = time.time()

    # Run Ollama translation with glossary support
    ollama_translation, ollama_time = translate_ollama(
        text, source_lang, target_lang, context_history,
        glossary_manager=glossary_manager
    )

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

    # Initialize the glossary manager
    glossary_mgr = get_glossary_manager()
    try:
        glossary_mgr.load_all()
    except Exception as e:
        logger.warning(f"Failed to load glossaries: {e}. Continuing without glossary support.")

    # Pre-warm Ollama to load the model into memory
    try:
        # Check if model exists, download if it doesn't
        logger.info(f"Checking if model {LLM_MODEL} is available locally...")
        try:
            ollama.show(LLM_MODEL)
        except ollama.ResponseError as e:
            if e.status_code == 404:
                logger.info(f"Model {LLM_MODEL} not found. Downloading (this may take a while)...")
                ui_queue.put({"type": "status", "process": "translator", "status": f"Downloading {LLM_MODEL}..."})
                ollama.pull(LLM_MODEL)
                logger.info(f"Model {LLM_MODEL} downloaded successfully.")
            else:
                raise e

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

                # Submit to thread pool with glossary manager
                executor.submit(
                    process_translation_task,
                    (text, lang), context_history.copy(), ui_queue, timing,
                    glossary_mgr
                )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in translator loop: {e}")
                ui_queue.put({"type": "error", "message": f"Translation Error: {e}"})
