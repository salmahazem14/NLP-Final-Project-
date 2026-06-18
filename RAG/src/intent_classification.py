from groq import Groq
import os
from RAG.src.config import settings
VALID_LABELS = {
    "greeting",
    "goodbye",
    "gratitude",
    "asking_mental_health_question",
    "out_of_scope"
}


client = Groq(api_key=settings.groq_api_key)

def build_prompt(user_input):
    return f"""
You are a strict intent classifier.

Labels:
- greeting
- goodbye
- gratitude
- asking_mental_health_question
- out_of_scope

Definitions:

greeting:
Simple greetings only.
Examples:
- hi
- hello
- hey

goodbye:
Ending conversation.
Examples:
- bye
- goodbye
- see you

gratitude:
Expressing thanks only.
Examples:
- thanks
- thank you

asking_mental_health_question:
User expresses emotional distress, mental suffering,
or asks for emotional/mental help.
Examples:
- I feel depressed
- I am anxious
- I feel hopeless
- I want to die
- can you help with stress?

out_of_scope:
Everything else:
- coding
- math
- factual questions
- neutral statements
- sarcasm without explicit distress

Rules:
- Return ONE label only
- No punctuation
- No explanations
- If explicit emotional distress exists, choose asking_mental_health_question
- If goodbye exists with emotional distress, choose asking_mental_health_question
- Be conservative about mental health classification
- Do not infer hidden emotions unless clearly stated

Input: "{user_input}"

Output:
"""


def classify_intent(user_input):
    prompt = build_prompt(user_input)
    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages =[{"role": "user", "content": prompt}],
        temperature = 0
    )
    return response.choices[0].message.content.strip()


def safe_classify_intent(user_input):
    intent = classify_intent(user_input).lower()

    if intent not in VALID_LABELS:
        return "out_of_scope"
    
    return intent 


if __name__ == "__main__":
    test_inputs = [ 
        "Hi there", 
        "I feel really depressed lately", 
        "Thanks!!", 
        "Can you help me code in Python?", 
        "Goodbye", "Hi , Bye" , 
        "I am so grateful for your help, but I have to go now. Goodbye!",
        "Hi I feel anxious", "Thanks, I’ve been depressed for weeks", 
        "Hello, can you help me with my stress?", 
        "I don’t know what I’m feeling",
        "I’m fine I guess…",
        "Hey I’ve been feeling stressed", 
        "Thanks, I feel like I want to die", 
        "I have to go but I feel anxious", 
        "How do I train a machine learning model?", 
        "What is depression in biology", 
        "Hey… I don’t even know… I’m tired of everything",
        "Hi, thanks, bye", "Yeah everything is just great…", 
        "I love being anxious all the time 🙃", 
        "مرحبا أنا مكتئب" ] 
    
    for text in test_inputs: print(text, "->", safe_classify_intent(text))