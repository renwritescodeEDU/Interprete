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

TRANSLATION_PROMPT_TEMPLATE = """\
<task>Simultaneous interpreter for live professional calls. {source_lang} → {target_lang}.</task>

<rules>
REGISTER: Always use formal address. In Spanish: "usted/su/le/él/ella" NEVER "tú/tu/te".
COMPLETENESS: Translate EVERY word. Never omit, summarize, or paraphrase. Keep all numbers, dates, codes, phone numbers unchanged.
ACCURACY: Translate compound terms correctly. Examples: mother-in-law=suegra, checking account=cuenta corriente, child support=manutención de menores, food stamps=cupones de alimento (SNAP), alley=callejón, dumpster=contenedor de basura, non-payment=falta de pago / impago.
ACRONYMS: First use → expand with Spanish meaning in parentheses. Example: APR → TAP (Tasa Anual de Porcentaje).
SPEAKER NOTE: {speaker_note}
</rules>

{glossary_section}

<context>
{context_str}
</context>

<text_to_translate>
{text}
</text_to_translate>

Output ONLY valid JSON with key "translation". No extra text."""


def _detect_same_language(text: str, target_lang: str) -> bool:
    """
    Quick heuristic: if we're supposed to translate TO Spanish but the text
    looks Spanish, or vice versa, skip translation to prevent re-translations.
    Uses high-frequency function words as a signal.
    """
    text_lower = text.lower()
    spanish_markers = {"el ", "la ", "los ", "las ", "que ", "de ", "en ", "con ", " es ", " y ", " no ", " un ", " una "}
    english_markers = {"the ", "and ", "to ", "of ", "is ", "in ", "that ", "it ", "for ", "was "}

    spanish_score = sum(1 for m in spanish_markers if m in text_lower)
    english_score = sum(1 for m in english_markers if m in text_lower)

    detected_is_spanish = spanish_score > english_score

    if target_lang == "Spanish" and detected_is_spanish:
        return True   # text is already Spanish, don't re-translate
    if target_lang == "English" and not detected_is_spanish and english_score > 0:
        return True   # text is already English, don't re-translate
    return False




def translate_ollama(text: str, source_lang: str, target_lang: str,
                     context_history: list, glossary_manager=None,
                     speaker_note: str = "Unknown speaker.") -> tuple:
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
        speaker_note=speaker_note,
        glossary_section=glossary_section if glossary_section else "No specific terminology loaded.",
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
                             glossary_manager=None, speaker_note: str = "Unknown speaker."):
    """Processes a single translation task using Ollama. Propagates pipeline timing."""
    text, lang = task[:2]

    target_lang = "Spanish" if lang == "en" else "English"
    source_lang = "English" if lang == "en" else "Spanish"

    # Guard: if the text is already in the target language, skip translation
    if _detect_same_language(text, target_lang):
        logger.info(f"[TRANSLATOR] Skipping re-translation — text already in {target_lang}.")
        return

    timing["translation_start"] = time.time()

    # Run Ollama translation with glossary support
    ollama_translation, ollama_time = translate_ollama(
        text, source_lang, target_lang, context_history,
        glossary_manager=glossary_manager,
        speaker_note=speaker_note
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
                    glossary_mgr, "Agent or client speaking — use formal register (usted)."
                )


            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in translator loop: {e}")
                ui_queue.put({"type": "error", "message": f"Translation Error: {e}"})
