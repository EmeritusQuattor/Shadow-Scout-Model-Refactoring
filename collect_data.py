import requests
import json
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.elo import EloSystem

GQL = 'https://api.ftcscout.org/graphql'
SEASON = 2025
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

session = requests.Session()
retry = Retry(total=5, backoff_factor=2)
adapter = HTTPAdapter(max_retries=retry)
session.mount('https://', adapter)

elo = EloSystem(k=32)


def gql(query):
    res = session.post(GQL, json={'query': query}, timeout=30)
    if not res.ok:
        print(f'  Response body: {res.text[:500]}')
    res.raise_for_status()
    data = res.json()
    if 'errors' in data:
        raise Exception(data['errors'][0]['message'])
    return data['data']


def fetch_events():
    data = gql(f'''{{
        eventsSearch(season: {SEASON}) {{
            code
            name
            type
        }}
    }}''')
    return data['eventsSearch']


def fetch_event_matches(event_code):
    data = gql(f'''{{
        eventByCode(code: "{event_code}", season: {SEASON}) {{
            matches {{
                id
                matchNum
                tournamentLevel
                teams {{
                    teamNumber
                    alliance
                    station
                }}
                scores {{
                    ... on MatchScores2025 {{
                        red {{ totalPoints autoPoints dcPoints totalPointsNp }}
                        blue {{ totalPoints autoPoints dcPoints totalPointsNp }}
                    }}
                }}
            }}
        }}
    }}''')
    return data.get('eventByCode', {}).get('matches', [])


def fetch_team_stats(team_number):
    data = gql(f'''{{
        teamByNumber(number: {team_number}) {{
            quickStats(season: {SEASON}) {{
                tot  {{ value rank }}
                auto {{ value rank }}
                dc   {{ value rank }}
                eg   {{ value rank }}
            }}
        }}
    }}''')
    return data.get('teamByNumber', {}).get('quickStats', None)


def extract_match_features(matches):
    features = []
    matches_sorted = sorted(matches, key=lambda m: m.get('matchNum', 0))

    for m in matches_sorted:
        scores = m.get('scores')
        if not scores:
            continue

        teams = m.get('teams', [])
        red_teams = sorted([t['teamNumber'] for t in teams if t['alliance'] == 'Red'])
        blue_teams = sorted([t['teamNumber'] for t in teams if t['alliance'] == 'Blue'])

        if len(red_teams) < 2 or len(blue_teams) < 2:
            continue

        red  = scores.get('red', {})
        blue = scores.get('blue', {})

        red_score  = red.get('totalPoints', 0)
        blue_score = blue.get('totalPoints', 0)

        if red_score == 0 and blue_score == 0:
            continue

        red_elo  = sum(elo.get_rating(t) for t in red_teams) / len(red_teams)
        blue_elo = sum(elo.get_rating(t) for t in blue_teams) / len(blue_teams)

        elo.update_match(red_teams, blue_teams, red_score, blue_score)

        features.append({
            'match_id': m['id'],
            'match_num': m['matchNum'],
            'level': m.get('tournamentLevel', ''),
            'red_teams': red_teams,
            'blue_teams': blue_teams,
            'red_score': red_score,
            'blue_score': blue_score,
            'red_auto': red.get('autoPoints', 0),
            'blue_auto': blue.get('autoPoints', 0),
            'red_dc': red.get('dcPoints', 0),
            'blue_dc': blue.get('dcPoints', 0),
            'red_np': red.get('totalPointsNp', 0),
            'blue_np': blue.get('totalPointsNp', 0),
            'red_score_np': red.get('totalPointsNp', 0),
            'blue_score_np': blue.get('totalPointsNp', 0),
            'red_elo': round(red_elo, 1),
            'blue_elo': round(blue_elo, 1),
            'elo_diff': round(red_elo - blue_elo, 1),
            'red_won': 1 if red_score > blue_score else 0,
        })

    return features


def main():
    print('Fetching events...')
    events = fetch_events()
    print(f'Found {len(events)} events')

    all_match_features = []

    for i, event in enumerate(events[:200]):
        print(f'[{i+1}/30] {event["name"]}')
        try:
            matches = fetch_event_matches(event['code'])
            if not matches:
                continue

            features = extract_match_features(matches)
            all_match_features.extend(features)

            time.sleep(0.3)
        except Exception as e:
            print(f'  Error: {e}')
            continue

    team_numbers = set()
    for m in all_match_features:
        team_numbers.update(m['red_teams'])
        team_numbers.update(m['blue_teams'])

    print(f'\nCollected {len(all_match_features)} matches from {len(team_numbers)} unique teams')

    print('Fetching team stats...')
    team_stats = {}
    for j, num in enumerate(team_numbers):
        try:
            stats = fetch_team_stats(num)
            if stats:
                team_stats[str(num)] = stats
            if j % 50 == 0:
                print(f'  {j}/{len(team_numbers)} teams...')
            time.sleep(0.1)
        except Exception as e:
            print(f'  Error team {num}: {e}')

    output = {
        'matches': all_match_features,
        'team_stats': team_stats,
        'elo_ratings': elo.ratings,
    }

    raw_path = os.path.join(DATA_DIR, 'matches_raw.json')
    with open(raw_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\nDone. Saved {len(all_match_features)} matches and {len(team_stats)} teams to {raw_path}')
    print(f'ELO ratings computed for {len(elo.ratings)} teams')


if __name__ == '__main__':
    main()
