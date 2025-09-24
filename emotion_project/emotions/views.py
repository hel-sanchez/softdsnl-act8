from django.http import JsonResponse
import tensorflow as tf
import numpy as np
import pickle
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os

# Load model and preprocessing tools
model_path = os.path.join(settings.BASE_DIR, 'media', 'models', 'emotion_model.h5')
tokenizer_path = os.path.join(settings.BASE_DIR, 'media', 'models', 'tokenizer.pkl')
encoder_path = os.path.join(settings.BASE_DIR, 'media', 'models', 'encoder.pkl')

model = tf.keras.models.load_model(model_path)
tokenizer = pickle.load(open(tokenizer_path, "rb"))
encoder = pickle.load(open(encoder_path, "rb"))

# Debug: Print the encoder classes (emotion labels)
print("Encoder Classes:", encoder.classes_)

# Manually define the emotion labels (this should match the order used during training)
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral", "other"
]

@csrf_exempt  # Disable CSRF protection for this view
def predict_emotion(request):
    if request.method == "POST":
        import json
        body = json.loads(request.body)
        text = body.get("text", "")

        # Tokenize and pad the input text
        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=50)
        
        # Make the prediction
        prediction = model.predict(padded)
        
        # Get the predicted class index (the highest probability index)
        predicted_index = np.argmax(prediction)
        
        # Debug: Print emotion index
        print(f"Predicted Index: {predicted_index}")
        
        # Manually map the predicted index to the emotion label
        predicted_emotion = emotion_labels[predicted_index]

        # Debug: Print the emotion that corresponds to the predicted index
        print(f"Predicted Emotion: {predicted_emotion}")
        
        # Return the result as JSON
        return JsonResponse({"text": text, "predicted_emotion": predicted_emotion})
