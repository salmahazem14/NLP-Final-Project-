from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def translate(text, source_language, target_language):
    """
    Translate text from source language to target language using Groq LLM.
    
    Args:
        text (str): The text to translate
        source_language (str): Source language code (e.g., 'ar', 'en')
        target_language (str): Target language code (e.g., 'ar', 'en')
    
    Returns:
        str: Translated text, or original text if translation fails
    """
    if source_language.lower() == target_language.lower():
        return text
    
    prompt = f"""Translate the following text from {source_language} to {target_language}.
Only provide the translation, no explanations.

Text: {text}

Translation:"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip().strip('"').strip("'")
    
    except Exception as e:
        print(f"[Translation Error]: {e}")
        return text  