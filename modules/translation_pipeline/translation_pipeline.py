from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def translate(text, source_language, target_language):
    """
    Translate text from source language to target language using Groq LLM.
    
    Only called when source_language != target_language (handled by caller).
    
    Args:
        text (str): The text to translate
        source_language (str): The source language code (e.g., 'ar', 'fr', 'es', 'en')
        target_language (str): The target language code (e.g., 'ar', 'fr', 'es', 'en')
    
    Returns:
        str: Translated text
    
    Example:
        >>> translate("أنا قلق", "ar", "en")
        "I am anxious"
    """
    # If same language, return as-is
    if source_language.lower() == target_language.lower():
        return text
    
    prompt = f"""Translate from {source_language} to {target_language}.
Only provide the translation, no explanations.

Text: "{text}"

Translation:"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()
