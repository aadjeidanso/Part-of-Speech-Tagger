from flask import Flask, request, jsonify
from hmm import predict_pos
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data.get('sentence', '')
    # Log the received sentence
    print(f"Received sentence: {sentence}")
    
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400
    
    pos_tags = predict_pos(sentence)
    return jsonify({'sentence': sentence, 'pos_tags': pos_tags})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
