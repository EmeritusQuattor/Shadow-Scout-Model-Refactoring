import json
import os


def generate_strategies(match_data):
    strategies = []
    red_feat = match_data.get('red_feat', [0, 0, 0, 0])
    blue_feat = match_data.get('blue_feat', [0, 0, 0, 0])
    red_elo = match_data.get('red_elo', 1500)
    blue_elo = match_data.get('blue_elo', 1500)
    red_win_prob = match_data.get('red_win_prob', 50)
    blue_win_prob = match_data.get('blue_win_prob', 50)

    if red_feat[1] > blue_feat[1]:
        strategies.append({
            'type': 'auto',
            'text': 'Red has stronger autonomous — they should play aggressive in auto to build an early lead.',
            'priority': 'high',
        })
    else:
        strategies.append({
            'type': 'auto',
            'text': 'Blue has stronger autonomous — Red should prioritize a consistent, safe auto to stay close.',
            'priority': 'high',
        })

    if red_feat[2] > blue_feat[2]:
        strategies.append({
            'type': 'teleop',
            'text': 'Red dominates teleop stats — focus on fast cycle times and avoid penalties.',
            'priority': 'high',
        })
    else:
        strategies.append({
            'type': 'teleop',
            'text': 'Blue has the teleop edge — Red should play disruptive defense and capitalize on Blue mistakes.',
            'priority': 'high',
        })

    if red_feat[3] > blue_feat[3]:
        strategies.append({
            'type': 'endgame',
            'text': 'Red has stronger endgame — practice climb consistency to secure those points.',
            'priority': 'medium',
        })
    else:
        strategies.append({
            'type': 'endgame',
            'text': 'Blue excels in endgame — Red should ensure climb reliability and consider parking strategies.',
            'priority': 'medium',
        })

    elo_gap = abs(red_elo - blue_elo)
    if elo_gap < 30:
        strategies.append({
            'type': 'general',
            'text': 'Very evenly matched alliances — execution and penalty avoidance will decide this match.',
            'priority': 'high',
        })
    elif red_elo > blue_elo:
        strategies.append({
            'type': 'general',
            'text': f'Red has an ELO advantage ({red_elo:.0f} vs {blue_elo:.0f}) — play your game, avoid risky plays.',
            'priority': 'medium',
        })
    else:
        strategies.append({
            'type': 'general',
            'text': f'Red is the underdog ({red_elo:.0f} vs {blue_elo:.0f}) — take calculated risks and disrupt Blue rhythm.',
            'priority': 'medium',
        })

    return strategies


def generate_match_summary(match_data):
    red_teams = match_data.get('red_teams', [])
    blue_teams = match_data.get('blue_teams', [])
    red_score = match_data.get('red_score', 0)
    blue_score = match_data.get('blue_score', 0)
    red_win_prob = match_data.get('red_win_prob', 50)
    blue_win_prob = match_data.get('blue_win_prob', 50)

    if blue_score > red_score:
        winner = f"Blue ({', '.join(str(t) for t in blue_teams)})"
        loser = f"Red ({', '.join(str(t) for t in red_teams)})"
        winning_score = blue_score
        losing_score = red_score
    else:
        winner = f"Red ({', '.join(str(t) for t in red_teams)})"
        loser = f"Blue ({', '.join(str(t) for t in blue_teams)})"
        winning_score = red_score
        losing_score = blue_score

    score_diff = abs(red_score - blue_score)

    if score_diff <= 10:
        closeness = 'a nail-biter'
    elif score_diff <= 30:
        closeness = 'a competitive match'
    else:
        closeness = 'a dominant performance'

    summary = (
        f"Match prediction: {winner} wins {winning_score}-{losing_score} "
        f"({closeness}). "
        f"Red win probability: {red_win_prob:.0f}%, Blue: {blue_win_prob:.0f}%. "
    )

    return summary
