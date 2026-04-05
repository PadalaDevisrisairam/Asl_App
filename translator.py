from deep_translator import GoogleTranslator

def translate_sentence(text):
    return {
        "english": text,
        "hindi": GoogleTranslator(source='auto', target='hi').translate(text),
        "telugu": GoogleTranslator(source='auto', target='te').translate(text),
        "tamil": GoogleTranslator(source='auto', target='ta').translate(text),
        "kannada": GoogleTranslator(source='auto', target='kn').translate(text)
    }