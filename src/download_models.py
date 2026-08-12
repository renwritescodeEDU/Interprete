from transformers import pipeline

def download_models():
    print("Downloading English -> Spanish model (Helsinki-NLP/opus-mt-en-es)...")
    pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    
    print("Downloading Spanish -> English model (Helsinki-NLP/opus-mt-es-en)...")
    pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
    
    print("Models downloaded and cached successfully. The system is ready for offline use.")

if __name__ == "__main__":
    download_models()
