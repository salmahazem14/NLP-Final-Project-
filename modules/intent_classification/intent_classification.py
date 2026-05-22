from groq import Groq


VALID_LABELS = {
    "greeting",
    "goodbye",
    "gratitude",
    "asking_mental_health_question",
    "out_of_scope"
}


client = Groq(api_key="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")


def build_prompt(user_input):
    return f"""
You are an intent classification system.

Classify the user input into ONE of the following labels:
- greeting
- goodbye
- gratitude
- asking_mental_health_question
- out_of_scope

Rules:
- Output ONLY ONE label
- Only return the label.
- Do not explain.
- Be strict.
- Do NOT add punctuation
- Do NOT add extra text
- If message is about coding/programming → out_of_scope
- If message is about math → out_of_scope
- If message is about general knowledge → out_of_scope
- If message contains goodbye → ALWAYS output goodbye unless it is mental health or feelings related putput mental health intent
- If the message contains i guess or i think or maybe and a feeling or emotion → mental health intent
- Mental health intent ONLY if user is asking about feelings, emotions, anxiety, depression, stress

If multiple intents exist, choose the MOST IMPORTANT one:
Priority: asking_mental_health_question > goodbye > greeting > gratitude > out_of_scope

Examples:
Input: "Hello there"
Output: greeting

Input: "Thanks for your help"
Output: gratitude

Input: "I feel anxious all the time"
Output: asking_mental_health_question

Input: "What is 2+2?"
Output: out_of_scope

Now classify:
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