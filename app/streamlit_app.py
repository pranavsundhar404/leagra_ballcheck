import sys, os, time, cv2, numpy as np, matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st, pandas as pd
from core.yolo_pose import YoloPoseEstimator, write_video, detect_ball_release_frame
from core.analysis import analyze_sequence, summarize_df, find_horizontal_before, compute_extension_deg
from core.corrections import generate_corrections

st.set_page_config(page_title='LEAGRA BowlCheck — Ball Release Verdict', page_icon='🏏', layout='wide')
st.title('LEAGRA BowlCheck — Ball Release Verdict 🏏')
st.caption('Verdict based on elbow **at ball release** + extension Δ vs 15° rule.')

with st.expander('Instructions', expanded=True):
    st.markdown("""
- We detect **ball release** using YOLOv8 (sports ball class) near the bowler's wrist.
- We pick the **upper-arm horizontal** frame **before** release.
- We compute **Extension Δ = elbow@release − elbow@upper-arm-horizontal**.
- **Verdict**: ILLEGAL if elbow@release < 165° **or** Δ > 15°. (2D heuristic, training use only.)
""")

video_file = st.file_uploader('Upload bowling video', type=['mp4','mov','m4v'])
if video_file:
    st.video(video_file.read())

col_a,col_b,col_c = st.columns([1,1,1])
sample_every = col_a.slider('Process every Nth frame', 1, 5, 1)
right_arm = col_b.selectbox('Bowling arm', ['Right','Left'])
conf = col_c.slider('YOLO confidence', 0.1, 0.7, 0.25, 0.05)

run_btn = st.button('Analyze', type='primary')
outputs_dir = os.path.join('assets','outputs'); os.makedirs(outputs_dir, exist_ok=True)

if run_btn:
    if not video_file:
        st.error('Please upload a video first.'); st.stop()

    raw_path = os.path.join(outputs_dir, f'input_{int(time.time())}.mp4')
    with open(raw_path, 'wb') as f:
        f.write(video_file.getbuffer())

    with st.spinner('Running YOLOv8 Pose...'):
        est = YoloPoseEstimator(model_name='yolov8n-pose.pt', device=None)
        kp_seq, frames, meta, wrists = est.process_video(raw_path, sample_every=sample_every, conf=conf, right_arm=(right_arm=='Right'))

    if not kp_seq:
        st.error('No bowler detected. Use a clearer side-view.'); st.stop()

    # Detect ball release (rough) using general YOLOv8 model (sports ball)
    with st.spinner('Detecting ball release frame...'):
        release_idx = detect_ball_release_frame(raw_path, wrists, model_name='yolov8n.pt', conf=0.25)

    df = analyze_sequence(kp_seq, right_arm=(right_arm=='Right'))
    summary = summarize_df(df)

    # upper-arm horizontal before release
    horiz_idx = find_horizontal_before(df, release_idx)

    pre_angle, rel_angle, extension = compute_extension_deg(df, release_idx, horiz_idx)
    illegal_at_release = df.loc[release_idx, 'elbow_deg'] < 165.0
    illegal_by_delta = extension > 15.0
    legal = not (illegal_at_release or illegal_by_delta)

    verdict_text = 'LEGAL DELIVERY ✅' if legal else 'ILLEGAL DELIVERY ⚠️'
    verdict_color = '#12B886' if legal else '#E03131'

    badge_html = (
        f"<div style='padding:16px;border-radius:12px;background:{verdict_color}22;border:1px solid {verdict_color};'>"
        f"<div style='font-size:20px;font-weight:700;color:{verdict_color};margin-bottom:6px;'>Verdict: {verdict_text}</div>"
        f"<div style='color:#E6F1FF;'>Elbow @ Release: <b>{rel_angle:.1f}°</b> &nbsp; | &nbsp; Δ Extension: <b>{extension:.1f}°</b> &nbsp; | &nbsp; Threshold: <b>15.0°</b></div>"
        f"</div>"
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    # Overlay 'BALL RELEASE 🎯' on release frame thumbnail
    st.subheader('Key Frames (Ball Release)')
    pre_rgb = cv2.cvtColor(frames[horiz_idx], cv2.COLOR_BGR2RGB)
    release_rgb = cv2.cvtColor(frames[release_idx], cv2.COLOR_BGR2RGB)
    # draw label on release frame
    rr = release_rgb.copy()
    cv2.rectangle(rr, (14,14), (260,58), (0,0,0), -1)
    cv2.putText(rr, 'BALL RELEASE \xf0\x9f\x8e\xaf', (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
    c1,c2 = st.columns(2)
    c1.image(pre_rgb, caption=f'Upper-arm horizontal — Frame {horiz_idx} | Elbow {pre_angle:.1f}°', use_container_width=True)
    c2.image(rr, caption=f'Ball Release — Frame {release_idx} | Elbow {rel_angle:.1f}°', use_container_width=True)

    # Save CSV and annotated video (we'll also mark the release frame in the video by coloring text yellow that frame)
    # Re-encode annotated frames with a yellow label on release
    anno_frames = []
    for i, f in enumerate(frames):
        fr = f.copy()
        if i == release_idx:
            cv2.putText(fr, 'BALL RELEASE', (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
        anno_frames.append(fr)

    csv_path = os.path.join(outputs_dir, f'metrics_{int(time.time())}.csv'); df.to_csv(csv_path, index=False)
    vid_path = os.path.join(outputs_dir, f'annotated_{int(time.time())}.mp4'); write_video(anno_frames, vid_path, fps=meta[2])

    # Quick timeline + plot
    st.subheader('Timeline (Legal vs Suspect by elbow-angle heuristic)')
    status = (~df['suspect_illegal']).astype(int)
    fig2 = plt.figure(figsize=(8, 1.2)); plt.imshow(status.values[np.newaxis, :], aspect='auto', vmin=0, vmax=1); plt.yticks([]); plt.xlabel('Frame →')
    st.pyplot(fig2)

    st.subheader('Elbow Angle Across Frames')
    fig = plt.figure(); plt.plot(df['frame'], df['elbow_deg'], label='Elbow angle (deg)'); plt.axhline(165, linestyle='--', label='165° (15° limit)'); plt.xlabel('Frame'); plt.ylabel('Degrees'); plt.legend()
    st.pyplot(fig)

    st.subheader('Guidelines & Coaching Corrections')
    st.markdown('**Rule focus:** illegal if elbow **extends >15°** between **upper-arm horizontal** and **ball release**, OR if elbow is already <165° at release. (2D heuristic)')
    st.markdown(generate_corrections(summary, verdict_text=verdict_text))

    st.download_button('Download metrics CSV', data=open(csv_path, 'rb').read(), file_name=os.path.basename(csv_path))

    st.subheader('Annotated Video (Per-frame angles + RELEASE marker)')
    with open(vid_path, 'rb') as f:
        st.video(f.read())
