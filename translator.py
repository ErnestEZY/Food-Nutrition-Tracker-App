import streamlit as st
from deep_translator import GoogleTranslator

# Cache to store translations and avoid repeated API calls
translation_cache = {}

def translate_text(text, target_lang="en"):
    """
    Translate text to the target language using deep-translator.
    Caches results to improve performance.
    """
    # If target language is English, return the original text (no translation needed)
    if target_lang == "en":
        return text

    # Create a cache key based on text and target language
    cache_key = f"{text}_{target_lang}"
    
    # Check if the translation is already in the cache
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    try:
        # Translate using GoogleTranslator
        translated = GoogleTranslator(source='en', target=target_lang).translate(text)
        # Store in cache
        translation_cache[cache_key] = translated
        return translated
    except Exception as e:
        # Fallback to original text if translation fails
        st.warning(f"Translation failed: {e}. Displaying original text.")
        return text

def get_text(text):
    """
    Get the translated text based on the selected language in session state.
    """
    language = st.session_state.get("language", "en")
    # Map language codes to GoogleTranslator target codes
    lang_map = {
        "en": "en",
        "zh": "zh-CN",  # Chinese (Mandarin)
        "ms": "ms"      # Malay
    }
    target_lang = lang_map.get(language, "en")
    return translate_text(text, target_lang)