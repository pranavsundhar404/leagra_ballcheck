import numpy as np, pandas as pd
ELBOW_LIMIT_DEG=15.0

def analyze_sequence(kp_seq, right_arm=True) -> pd.DataFrame:
    from .yolo_pose import compute_angles, upper_arm_angle_to_horizontal
    rows=[]; upper=[]
    for i,kp in enumerate(kp_seq):
        e,s,t=compute_angles(kp,right_arm=right_arm)
        rows.append({'frame':i,'elbow_deg':e,'shoulder_tilt_deg':s,'trunk_lean_deg':t})
        upper.append(upper_arm_angle_to_horizontal(kp,right_arm=right_arm))
    df=pd.DataFrame(rows)
    df['upper_arm_to_horizontal_deg']=upper
    df['suspect_illegal']=df['elbow_deg']<(180.0-ELBOW_LIMIT_DEG)
    return df

def find_horizontal_before(df, release_idx):
    window=df[df['frame']<=release_idx]
    if len(window)==0: window=df
    horiz_idx=int((window['upper_arm_to_horizontal_deg'].abs()).idxmin())
    return horiz_idx

def compute_extension_deg(df, release_idx, horiz_idx):
    pre=float(df.loc[horiz_idx,'elbow_deg'])
    rel=float(df.loc[release_idx,'elbow_deg'])
    delta=max(0.0, rel-pre)
    return pre, rel, delta

def summarize_df(df: pd.DataFrame):
    out={'min_elbow_deg':float(np.nanmin(df['elbow_deg'])) if len(df) else np.nan,
         'mean_elbow_deg':float(np.nanmean(df['elbow_deg'])) if len(df) else np.nan,
         'max_trunk_lean_deg':float(np.nanmax(df['trunk_lean_deg'])) if len(df) else np.nan,
         'mean_shoulder_tilt_deg':float(np.nanmean(df['shoulder_tilt_deg'])) if len(df) else np.nan,
         'frames':int(len(df)),
         'illegal_frames_abs_elbow':int(df['suspect_illegal'].sum()) if len(df) else 0}
    return out
