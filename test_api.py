import sys
sys.path.insert(0, r'C:\Users\guill\OneDrive\Desktop\Scout-AI-Training')

from collect_data import fetch_events, fetch_event_matches
from utils.elo import EloSystem

events = fetch_events()
print(f'Events found: {len(events)}')

if events:
    e = events[0]
    print(f'First event: {e["name"]} ({e["code"]})')
    matches = fetch_event_matches(e['code'])
    print(f'Matches in {e["code"]}: {len(matches)}')
    if matches:
        first = matches[0]
        print(f'Sample match: matchNum={first.get("matchNum")}')
        scores = first.get('scores', {})
        if scores:
            print(f'Red score keys: {list(scores.get("red", {}).keys())}')
            print(f'Blue score keys: {list(scores.get("blue", {}).keys())}')
