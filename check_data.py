import json
d = json.load(open(r'C:\Users\guill\OneDrive\Desktop\Scout-AI-Training\data\matches_raw.json'))
print(f'Matches: {len(d["matches"])}')
print(f'Teams: {len(d["team_stats"])}')
print(f'ELOs: {len(d["elo_ratings"])}')
