import numpy as np
from collections import defaultdict


def compute_opr(matches):
    team_to_idx = {}
    equations = []

    for m in matches:
        for team in m['red_teams'] + m['blue_teams']:
            if team not in team_to_idx:
                team_to_idx[team] = len(team_to_idx)

    n_teams = len(team_to_idx)
    n_matches = len(matches)

    for m in matches:
        row_red = np.zeros(n_teams)
        for t in m['red_teams']:
            row_red[team_to_idx[t]] = 1.0
        equations.append((row_red, m['red_score']))

        row_blue = np.zeros(n_teams)
        for t in m['blue_teams']:
            row_blue[team_to_idx[t]] = 1.0
        equations.append((row_blue, m['blue_score']))

    A = np.array([e[0] for e in equations])
    b = np.array([e[1] for e in equations])

    opr, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    team_opr = {}
    for team, idx in team_to_idx.items():
        team_opr[team] = round(opr[idx], 2)

    return team_opr


def compute_dpr(matches):
    """
    DPR (Defensive Power Rating): how many points a team's alliance concedes when they play.
    Lower is better (they play good defense or their opponents score less).
    """
    team_to_idx = {}
    equations = []

    for m in matches:
        for team in m['red_teams'] + m['blue_teams']:
            if team not in team_to_idx:
                team_to_idx[team] = len(team_to_idx)

    n_teams = len(team_to_idx)

    for m in matches:
        row_red = np.zeros(n_teams)
        for t in m['red_teams']:
            row_red[team_to_idx[t]] = 1.0
        equations.append((row_red, m['blue_score']))

        row_blue = np.zeros(n_teams)
        for t in m['blue_teams']:
            row_blue[team_to_idx[t]] = 1.0
        equations.append((row_blue, m['red_score']))

    A = np.array([e[0] for e in equations])
    b = np.array([e[1] for e in equations])

    dpr, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    team_dpr = {}
    for team, idx in team_to_idx.items():
        team_dpr[team] = round(dpr[idx], 2)

    return team_dpr


def compute_opr_dpr(matches):
    return compute_opr(matches), compute_dpr(matches)
