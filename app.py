from flask import Flask, render_template, request, jsonify
import keras
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = keras.models.load_model('vietnam_food_class.keras')

# Classes from training (must match exact order from notebook)
CLASSES = ['Pho', 'Hu tieu', 'Bun bo Hue']

# Food descriptions
FOOD_INFO = {
    'Pho': {
        'vietnamese': 'Phở',
        'description': 'A Vietnamese soup consisting of broth, rice noodles, herbs, and meat (usually beef or chicken).',
        'origin': 'Northern Vietnam'
    },
    'Hu tieu': {
        'vietnamese': 'Hủ Tiếu',
        'description': 'A popular noodle soup from Southern Vietnam, typically made with pork, shrimp, and clear broth.',
        'origin': 'Southern Vietnam'
    },
    'Bun bo Hue': {
        'vietnamese': 'Bún Bò Huế',
        'description': 'A spicy beef noodle soup originating from Hue, featuring lemongrass and fermented shrimp paste.',
        'origin': 'Central Vietnam (Hue)'
    }
}

def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def index():
    return render_template('index.html', classes=CLASSES, food_info=FOOD_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        predicted_class = CLASSES[predicted_class_idx]
        food_details = FOOD_INFO.get(predicted_class, {})
        
        # Get all predictions sorted by confidence
        all_predictions = []
        for idx, conf in enumerate(predictions[0]):
            all_predictions.append({
                'class': CLASSES[idx],
                'vietnamese': FOOD_INFO[CLASSES[idx]]['vietnamese'],
                'confidence': float(conf) * 100
            })
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'vietnamese_name': food_details.get('vietnamese', ''),
            'description': food_details.get('description', ''),
            'origin': food_details.get('origin', ''),
            'confidence': confidence,
            'all_predictions': all_predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🍜 Vietnamese Food Classifier")
    print("=" * 40)
    print(f"Model: vietnam_food_class.keras")
    print(f"Classes: {', '.join(CLASSES)}")
    print("=" * 40)
    print("\n🌐 Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)
