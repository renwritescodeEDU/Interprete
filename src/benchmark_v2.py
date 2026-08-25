import time
import json
import ollama
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Qwen2.5 1.5B is our winning architecture
LLM_MODEL = "qwen2.5:1.5b"

# Exhaustive Edge Cases
TEST_CASES = [
    # 1. Extreme Medical Jargon
    {"category": "Medical Jargon", "lang": "es", "text": "Paciente presenta disnea de esfuerzo, ortopnea y edema bilateral de miembros inferiores con fóvea cruz triple, sugerente de insuficiencia cardíaca congestiva descompensada."},
    
    # 2. Tech / Legal
    {"category": "Legal/Tech", "lang": "en", "text": "Pursuant to section 4(b) of the Non-Disclosure Agreement, any unauthorized dissemination of proprietary cryptographic algorithms will result in immediate injunctive relief."},
    
    # 3. Sarcasm and Idioms
    {"category": "Idioms", "lang": "en", "text": "Oh great, it's raining cats and dogs right when I washed my car. This is just the icing on the cake, isn't it?"},
    {"category": "Slang (LatAm)", "lang": "es", "text": "Ese man es un chanta, me vendió un carro chatarra que me costó un ojo de la cara y a la semana se dañó el motor. ¡Qué vaina tan berraca!"},
    {"category": "Slang (Spain)", "lang": "es", "text": "Tío, me cago en la leche, he pillado un atasco monumental y para colmo me he dejado el móvil en el curro. ¡Menudo marrón!"},
    
    # 4. Rapid-fire Short Sentences (Latency test)
    {"category": "Rapid-fire", "lang": "en", "text": "Hello."},
    {"category": "Rapid-fire", "lang": "es", "text": "Tengo hambre."},
    {"category": "Rapid-fire", "lang": "en", "text": "No way!"},
    
    # 5. Run-on sentence without punctuation (Testing our critical instruction)
    {"category": "No Punctuation", "lang": "es", "text": "hola amigo como estas espero que muy bien te cuento que ayer fui al mercado compre peras manzanas y tomates pero se me olvido la billetera imaginate que verguenza pase con la señora de la tienda"},
    
    # 6. Ambiguous Context (Needs reasoning)
    {"category": "Ambiguity", "lang": "es", "text": "El banco estaba cerrado, así que me senté en el banco a esperar."}, # Bank vs Bench
    {"category": "Ambiguity", "lang": "en", "text": "I saw a man with a telescope."}, # Who has the telescope?
]

def benchmark_qwen():
    logger.info(f"\n--- Starting Benchmark V2 on {LLM_MODEL} ---")
    
    # Warmup
    try:
        ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': '{"test":"hi"}'}], format='json', options={'temperature': 0.0})
    except Exception as e:
        logger.error(f"Failed to load {LLM_MODEL}. ({e})")
        return

    results = []

    for idx, item in enumerate(TEST_CASES):
        cat = item["category"]
        lang = item["lang"]
        text = item["text"]
        
        target_lang = "Spanish" if lang == "en" else "English"
        source_lang = "English" if lang == "en" else "Spanish"

        prompt = f"""You are a strict JSON translation API. 
Your ONLY task is to translate the following {source_lang} text to {target_lang}.
Output ONLY valid JSON in this exact format: {{"translation": "the translated string"}}.
Do not add any conversational text, greetings, or explanations.

CRITICAL INSTRUCTION 1: Ensure the translation is perfectly punctuated with commas and periods to make it easy to read aloud smoothly. Do not output run-on sentences.
CRITICAL INSTRUCTION 2: If you encounter specialized medical or legal terminology that you do not understand, translate it literally. DO NOT invent or hallucinate medical terms.
CRITICAL INSTRUCTION 3: Translate colloquialisms, idioms, and slang into their natural cultural equivalent meaning, not literally (e.g. Spanish 'me cago en la leche' means 'damn it', 'chanta' means 'scammer').

Text to translate:
{text}
"""
        start_t = time.time()
        try:
            response = ollama.chat(
                model=LLM_MODEL, 
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.0}
            )
            res_json = json.loads(response['message']['content'])
            res = res_json.get('translation', 'Error')
        except Exception as e:
            res = f"Error: {e}"

        latency = time.time() - start_t
        
        logger.info(f"\n[{cat}] Latency: {latency:.2f}s")
        logger.info(f"Original: {text}")
        logger.info(f"Translated: {res}")
        
        results.append({
            "category": cat,
            "latency": latency,
            "original": text,
            "translated": res
        })

    # Summary
    avg_latency = sum(r["latency"] for r in results) / len(results)
    logger.info(f"\n--- Benchmark Summary ---")
    logger.info(f"Total phrases tested: {len(results)}")
    logger.info(f"Average Latency: {avg_latency:.2f}s")

if __name__ == "__main__":
    benchmark_qwen()
