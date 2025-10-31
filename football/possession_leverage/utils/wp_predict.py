import joblib
import numpy as np
import pandas as pd

# Load trained model (and, if available, saved feature list)
_model = joblib.load('../models/wp_model.pkl')

_FEAT_COLS = [
    'qtr','game_seconds_remaining','log_secs','score_differential','yardline_100',
    'down','ydstogo','posteam_is_home',
    'posteam_timeouts_remaining','defteam_timeouts_remaining',
    'abs_score_diff','two_possession_offense',
    'u120','u60','u30','u15'
]

def _build_post_try_rows(row: pd.Series, base_margin: int) -> pd.DataFrame:
    """
    Build the THREE post-try states (fail/base, XP+1, 2PT+2) from the leader's perspective.
    After the try, opponent has the ball. Model expects OFFENSE (posteam) perspective inputs.
    """
    qtr  = int(row['qtr'])
    secs = int(row['game_seconds_remaining'])  # GLOBAL seconds remaining

    # Leader margins to evaluate; convert to offense (opponent) perspective by negating
    leader_margins = [base_margin, base_margin + 1, base_margin + 2]
    posteam_diffs  = [-m for m in leader_margins]  # offense perspective

    # Neutralized post-try possession: opponent ball at own 25, 1st & 10
    yl, down, togo = 75, 1, 10

    # Timeouts: offense is opponent after try; defense is leader
    off_to = int(row.get('defteam_timeouts_remaining', 2))
    def_to = int(row.get('posteam_timeouts_remaining', 2))

    # Home flag for the POST-TRY OFFENSE (opponent). If you have a precomputed flag, use it; else default 0.
    is_home = int(row.get('posteam_is_home_after', 0))

    # Late-game features & transforms
    log_secs = float(np.log1p(secs))
    u120 = int(secs <= 120)
    u60  = int(secs <= 60)
    u30  = int(secs <= 30)
    u15  = int(secs <= 15)

    rows = []
    names = ['wp_fail', 'wp_xp_good', 'wp_2pt_good']  # order aligned with leader_margins

    for name, off_diff in zip(names, posteam_diffs):
        rows.append({
            'branch': name,
            'qtr': qtr,
            'game_seconds_remaining': secs,
            'log_secs': log_secs,
            'score_differential': off_diff,                 # offense perspective
            'yardline_100': yl,
            'down': down,
            'ydstogo': togo,
            'posteam_is_home': is_home,
            'posteam_timeouts_remaining': off_to,
            'defteam_timeouts_remaining': def_to,
            'abs_score_diff': abs(off_diff),                # derived from offense diff
            'two_possession_offense': int(off_diff >= 9),   # offense up >= 9
            'u120': u120, 'u60': u60, 'u30': u30, 'u15': u15
        })

    df = pd.DataFrame(rows)
    # Ensure exact training column order for predict
    return df[_FEAT_COLS + ['branch']]

def predict_wps(row: pd.Series, base_margin: int) -> pd.Series:
    """
    Returns a Series with keys: 'wp_fail', 'wp_xp_good', 'wp_2pt_good' (leader perspective).
    """
    states = _build_post_try_rows(row, base_margin)
    feats  = states[_FEAT_COLS]
    offense_wp = _model.predict(feats)  # model predicts offense (opponent) WP
    leader_wp  = 1.0 - offense_wp                  # flip to leader WP
    leader_wp = np.minimum(leader_wp, 0.999999)
    return pd.Series(dict(zip(states['branch'], leader_wp)))
