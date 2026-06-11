import json
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, 'matches_raw.json')) as f:
    raw = json.load(f)

matches = raw['matches']
team_stats = raw['team_stats']

from utils.opr import compute_opr_dpr
team_opr, team_dpr = compute_opr_dpr(matches)

for team_str, opr_val in team_opr.items():
    ts = team_stats.get(str(team_str), None)
    if ts:
        ts['opr'] = {'value': opr_val}

for team_str, dpr_val in team_dpr.items():
    ts = team_stats.get(str(team_str), None)
    if ts:
        ts['dpr'] = {'value': dpr_val}

n_with_opr = sum(1 for v in team_stats.values() if 'opr' in v)
print(f'Computed OPR/DPR for {n_with_opr} teams')


ROLLING_WINDOW = 5


def compute_rolling_averages(match_data):
    team_history = {}
    match_data_sorted = sorted(match_data, key=lambda m: m.get('match_id', 0))

    for m in match_data_sorted:
        for team in m['red_teams'] + m['blue_teams']:
            if team not in team_history:
                team_history[team] = {'scores': [], 'autos': [], 'dcs': []}

        red_teams = m['red_teams']
        blue_teams = m['blue_teams']
        red_score = m['red_score']
        blue_score = m['blue_score']
        red_auto = m['red_auto']
        blue_auto = m['blue_auto']
        red_dc = m['red_dc']
        blue_dc = m['blue_dc']

        red_avg = []
        for t in red_teams:
            h = team_history[t]
            avg = (
                sum(h['scores'][-ROLLING_WINDOW:]) / max(len(h['scores'][-ROLLING_WINDOW:]), 1),
                sum(h['autos'][-ROLLING_WINDOW:]) / max(len(h['autos'][-ROLLING_WINDOW:]), 1),
                sum(h['dcs'][-ROLLING_WINDOW:]) / max(len(h['dcs'][-ROLLING_WINDOW:]), 1),
            )
            red_avg.append(avg)

        blue_avg = []
        for t in blue_teams:
            h = team_history[t]
            avg = (
                sum(h['scores'][-ROLLING_WINDOW:]) / max(len(h['scores'][-ROLLING_WINDOW:]), 1),
                sum(h['autos'][-ROLLING_WINDOW:]) / max(len(h['autos'][-ROLLING_WINDOW:]), 1),
                sum(h['dcs'][-ROLLING_WINDOW:]) / max(len(h['dcs'][-ROLLING_WINDOW:]), 1),
            )
            blue_avg.append(avg)

        for t in red_teams:
            team_history[t]['scores'].append(red_score)
            team_history[t]['autos'].append(red_auto)
            team_history[t]['dcs'].append(red_dc)

        for t in blue_teams:
            team_history[t]['scores'].append(blue_score)
            team_history[t]['autos'].append(blue_auto)
            team_history[t]['dcs'].append(blue_dc)

        avg_red_tot = sum(a[0] for a in red_avg)
        avg_red_auto = sum(a[1] for a in red_avg)
        avg_red_dc = sum(a[2] for a in red_avg)
        avg_blue_tot = sum(a[0] for a in blue_avg)
        avg_blue_auto = sum(a[1] for a in blue_avg)
        avg_blue_dc = sum(a[2] for a in blue_avg)

        m['roll_red_tot'] = round(avg_red_tot, 1)
        m['roll_red_auto'] = round(avg_red_auto, 1)
        m['roll_red_dc'] = round(avg_red_dc, 1)
        m['roll_blue_tot'] = round(avg_blue_tot, 1)
        m['roll_blue_auto'] = round(avg_blue_auto, 1)
        m['roll_blue_dc'] = round(avg_blue_dc, 1)


def team_to_features(t):
    s = t.get('stats') or {}
    return [
        s.get('tot', {}).get('value') or 0,
        s.get('auto', {}).get('value') or 0,
        s.get('dc', {}).get('value') or 0,
        s.get('eg', {}).get('value') or 0,
        s.get('opr', {}).get('value') or 0,
        s.get('dpr', {}).get('value') or 0,
    ]


def build_features(match_data):
    compute_rolling_averages(match_data)

    X_clf, y_clf = [], []
    X_reg, y_reg = [], []

    level_map = {'Quals': 0, 'Finals': 1, 'DoubleElim': 2}

    for m in match_data:
        red_teams = m['red_teams']
        blue_teams = m['blue_teams']

        if len(red_teams) < 2 or len(blue_teams) < 2:
            continue

        red_t_feat = [team_to_features({'stats': team_stats.get(str(t), {})}) for t in red_teams]
        blue_t_feat = [team_to_features({'stats': team_stats.get(str(b), {})}) for b in blue_teams]

        red_sum = [sum(t[i] for t in red_t_feat) for i in range(6)]
        blue_sum = [sum(t[i] for t in blue_t_feat) for i in range(6)]

        level_enc = level_map.get(m.get('level', 'Quals'), 0)
        is_quals = 1 if level_enc == 0 else 0
        is_finals = 1 if level_enc == 1 else 0
        is_doubleelim = 1 if level_enc == 2 else 0

        features = (
            red_sum + blue_sum
            + [m.get('red_elo', 1500), m.get('blue_elo', 1500), m.get('elo_diff', 0)]
            + [m.get('roll_red_tot', 0), m.get('roll_red_auto', 0), m.get('roll_red_dc', 0)]
            + [m.get('roll_blue_tot', 0), m.get('roll_blue_auto', 0), m.get('roll_blue_dc', 0)]
            + [
                red_sum[0] - blue_sum[0],
                red_sum[0] * blue_sum[0],
                red_sum[4] - blue_sum[4],
                red_sum[5] - blue_sum[5],
            ]
            + [is_quals, is_finals, is_doubleelim]
        )

        X_clf.append(features)
        y_clf.append(m['red_won'])

        X_reg.append(features)
        y_reg.append([m['red_score'], m['blue_score']])

    return np.array(X_clf), np.array(y_clf), np.array(X_reg), np.array(y_reg)


print('Building feature matrix...')
X_clf, y_clf, X_reg, y_reg = build_features(matches)
print(f'Samples: {len(X_clf)}')


X_train, X_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
_, _, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)


print('\n=== Classifier (Win Prediction) ===')
print('Tuning hyperparameters...')

clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}

grid_clf = GridSearchCV(
    clf, param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_clf.fit(X_train, yc_train)

print(f'Best params: {grid_clf.best_params_}')
print(f'Best CV accuracy: {grid_clf.best_score_:.2%}')

best_clf = grid_clf.best_estimator_
y_pred = best_clf.predict(X_test)
acc = accuracy_score(yc_test, y_pred)
print(f'Test accuracy: {acc:.2%}')
print('\nClassification Report:')
print(classification_report(yc_test, y_pred))


print('\n=== Regressor (Score Prediction) ===')
print('Tuning hyperparameters...')

reg = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    random_state=42,
    n_jobs=-1
)

param_grid_reg = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
}

grid_reg = GridSearchCV(
    reg, param_grid_reg,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_reg.fit(X_train, yr_train)

print(f'Best params: {grid_reg.best_params_}')
print(f'Best CV MAE: {-grid_reg.best_score_:.1f} pts')

best_reg = grid_reg.best_estimator_
y_pred_reg = best_reg.predict(X_test)
mae = mean_absolute_error(yr_test, y_pred_reg)
print(f'Test MAE: {mae:.1f} pts')


clf_path = os.path.join(MODEL_DIR, 'model_classifier.pkl')
reg_path = os.path.join(MODEL_DIR, 'model_regressor.pkl')

with open(clf_path, 'wb') as f:
    pickle.dump(best_clf, f)
with open(reg_path, 'wb') as f:
    pickle.dump(best_reg, f)

print(f'\nModels saved to {MODEL_DIR}')

feature_importance = best_clf.feature_importances_
feature_names = (
    ['red_tot', 'red_auto', 'red_dc', 'red_eg', 'red_opr', 'red_dpr']
    + ['blue_tot', 'blue_auto', 'blue_dc', 'blue_eg', 'blue_opr', 'blue_dpr']
    + ['red_elo', 'blue_elo', 'elo_diff']
    + ['roll_r_tot', 'roll_r_auto', 'roll_r_dc']
    + ['roll_b_tot', 'roll_b_auto', 'roll_b_dc']
    + ['tot_diff', 'tot_inter', 'opr_diff', 'dpr_diff']
    + ['is_quals', 'is_finals', 'is_doubleelim']
)
print('\nFeature Importance (Classifier):')
for name, imp in sorted(zip(feature_names, feature_importance), key=lambda x: -x[1]):
    print(f'  {name}: {imp:.3f}')


print('\n=== LogisticRegression Baseline ===')
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, yc_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(yc_test, lr_pred)
print(f'LogisticRegression test accuracy: {lr_acc:.2%}')
print(classification_report(yc_test, lr_pred))
