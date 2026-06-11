import pickle
import numpy as np
import os
from flask import Flask, request, jsonify
from generation.match_text import generate_strategies, generate_match_summary

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

app = Flask(__name__)

with open(os.path.join(MODEL_DIR, 'model_classifier.pkl'), 'rb') as f:
    clf = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'model_regressor.pkl'), 'rb') as f:
    reg = pickle.load(f)

print(f'Models loaded from {MODEL_DIR}')


def team_to_features(t):
    s = t.get('stats') or {}
    return [
        s.get('tot', {}).get('value') or 0,
        s.get('auto', {}).get('value') or 0,
        s.get('dc', {}).get('value') or 0,
        s.get('eg', {}).get('value') or 0,
    ]


def alliance_features(teams):
    f1 = team_to_features(teams[0])
    f2 = team_to_features(teams[1])
    return [f1[i] + f2[i] for i in range(4)]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    red = data.get('red', [])
    blue = data.get('blue', [])

    if len(red) < 2 or len(blue) < 2:
        return jsonify({'error': 'Need at least 2 teams per alliance'}), 400

    red_feat = alliance_features(red)
    blue_feat = alliance_features(blue)

    red_elo = sum(t.get('elo', 1500) for t in red) / len(red)
    blue_elo = sum(t.get('elo', 1500) for t in blue) / len(blue)

    features = np.array([red_feat + blue_feat + [red_elo, blue_elo, red_elo - blue_elo]])

    red_win_prob = float(clf.predict_proba(features)[0][1])
    blue_win_prob = 1.0 - red_win_prob

    scores = reg.predict(features)[0]
    red_score = round(float(scores[0]))
    blue_score = round(float(scores[1]))

    match_data = {
        'red_teams': [t.get('number') for t in red],
        'blue_teams': [t.get('number') for t in blue],
        'red_feat': red_feat,
        'blue_feat': blue_feat,
        'red_elo': red_elo,
        'blue_elo': blue_elo,
        'red_score': red_score,
        'blue_score': blue_score,
        'red_win_prob': red_win_prob * 100,
        'blue_win_prob': blue_win_prob * 100,
    }

    strategies = generate_strategies(match_data)
    summary = generate_match_summary(match_data)

    return jsonify({
        'predictions': {
            'red_win_prob': round(red_win_prob * 100, 1),
            'blue_win_prob': round(blue_win_prob * 100, 1),
            'red_score': red_score,
            'blue_score': blue_score,
        },
        'strategies': strategies,
        'summary': summary,
        'features': {
            'red': red_feat,
            'blue': blue_feat,
            'red_elo': round(red_elo, 1),
            'blue_elo': round(blue_elo, 1),
            'elo_diff': round(red_elo - blue_elo, 1),
        },
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models': ['classifier', 'regressor']})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
