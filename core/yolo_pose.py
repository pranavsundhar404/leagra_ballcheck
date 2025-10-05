from typing import Optional, List, Tuple
import numpy as np, cv2, os
from ultralytics import YOLO

# COCO keypoints indices for YOLOv8 Pose
KP={'L_SHOULDER':5,'R_SHOULDER':6,'L_ELBOW':7,'R_ELBOW':8,'L_WRIST':9,'R_WRIST':10,'L_HIP':11,'R_HIP':12}
ELBOW_LIMIT_DEG=15.0

def angle_3pts(a,b,c):
    a,b,c=np.array(a[:2],float),np.array(b[:2],float),np.array(c[:2],float)
    ba=a-b; bc=c-b
    denom=(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
    if denom==0: return np.nan
    cosang=np.dot(ba,bc)/denom
    cosang=np.clip(cosang,-1.0,1.0)
    return float(np.degrees(np.arccos(cosang)))

def shoulder_elbow_vector(kp,right_arm=True):
    if right_arm: s=kp[KP['R_SHOULDER'],:2]; e=kp[KP['R_ELBOW'],:2]
    else: s=kp[KP['L_SHOULDER'],:2]; e=kp[KP['L_ELBOW'],:2]
    return e-s

def upper_arm_angle_to_horizontal(kp,right_arm=True):
    v=shoulder_elbow_vector(kp,right_arm=right_arm)
    return float(np.degrees(np.arctan2(v[1],v[0]+1e-8)))

def compute_angles(kp,right_arm=True):
    if right_arm: shoulder=kp[KP['R_SHOULDER'],:2]; elbow=kp[KP['R_ELBOW'],:2]; wrist=kp[KP['R_WRIST'],:2]
    else: shoulder=kp[KP['L_SHOULDER'],:2]; elbow=kp[KP['L_ELBOW'],:2]; wrist=kp[KP['L_WRIST'],:2]
    elbow_angle=angle_3pts(shoulder,elbow,wrist)
    ls=kp[KP['L_SHOULDER'],:2]; rs=kp[KP['R_SHOULDER'],:2]
    dx=rs[0]-ls[0]; dy=rs[1]-ls[1]
    shoulder_tilt=float(np.degrees(np.arctan2(dy,dx+1e-8)))
    lh=kp[KP['L_HIP'],:2]; rh=kp[KP['R_HIP'],:2]
    mid_sh=(ls+rs)/2.0; mid_hip=(lh+rh)/2.0
    dx2=mid_sh[0]-mid_hip[0]; dy2=mid_sh[1]-mid_hip[1]
    trunk_lean=abs(float(np.degrees(np.arctan2(dx2,dy2+1e-8))))
    return elbow_angle,shoulder_tilt,trunk_lean

class YoloPoseEstimator:
    def __init__(self,model_name='yolov8n-pose.pt',device:Optional[str]=None):
        self.model=YOLO(model_name)
        if device is not None: self.model.to(device)

    def process_video(self,path,sample_every=1,conf=0.25,right_arm=True):
        cap=cv2.VideoCapture(path)
        if not cap.isOpened(): raise RuntimeError('Could not open video: '+path)
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        keypoints_seq=[]; annotated_frames=[]; wrists=[]; i=0
        while True:
            ret,frame=cap.read()
            if not ret: break
            if i%sample_every!=0: i+=1; continue
            res=self.model.predict(frame,verbose=False,conf=conf,imgsz=640)[0]
            if res.keypoints is None or res.boxes is None or len(res.boxes)==0: i+=1; continue
            boxes=res.boxes.xyxy.cpu().numpy()
            idx=int(np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])))
            xy=res.keypoints.xy.cpu().numpy(); kp=xy[idx]
            kp_pad=np.concatenate([kp,np.ones((kp.shape[0],1))],axis=1)
            keypoints_seq.append(kp_pad)

            # save wrist (both) for release proximity later
            l_wrist=kp[KP['L_WRIST']]; r_wrist=kp[KP['R_WRIST']]
            wrists.append((tuple(l_wrist),tuple(r_wrist)))

            annotated=res.plot()
            e,stilt,tlean=compute_angles(kp_pad,right_arm=right_arm)
            color=(0,255,0) if e>=165.0 else (0,0,255)
            text=f'Elbow: {e:.1f}° | Tilt: {stilt:.1f}° | Trunk: {tlean:.1f}°'
            cv2.rectangle(annotated,(14,14),(520,60),(0,0,0),-1)
            cv2.putText(annotated,text,(20,45),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2,cv2.LINE_AA)
            annotated_frames.append(annotated)
            i+=1
        cap.release()
        return keypoints_seq, annotated_frames, (width,height,fps,total), wrists

def detect_ball_release_frame(video_path, wrists, model_name='yolov8n.pt', conf=0.25):
    """Detect first frame where COCO 'sports ball' (class 32) appears near a wrist. Fallback to first ball sighting."""
    model=YOLO(model_name)
    cap=cv2.VideoCapture(video_path)
    frame_idx=0; release_idx=None; first_ball_idx=None
    # distance threshold relative to width for scale invariance
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); prox_thresh=max(20,int(width*0.12))
    while True:
        ret,frame=cap.read()
        if not ret: break
        res=model.predict(frame,conf=conf,verbose=False,classes=[32])[0]  # sports ball
        if res.boxes is not None and len(res.boxes)>0:
            if first_ball_idx is None: first_ball_idx=frame_idx
            # center of the largest ball box
            bb=res.boxes.xyxy.cpu().numpy()
            areas=(bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1])
            j=int(np.argmax(areas))
            cx=int((bb[j,0]+bb[j,2])/2); cy=int((bb[j,1]+bb[j,3])/2)
            if frame_idx < len(wrists):
                lw, rw = wrists[frame_idx]
                d1=np.hypot(cx-lw[0], cy-lw[1])
                d2=np.hypot(cx-rw[0], cy-rw[1])
                if min(d1,d2) <= prox_thresh:
                    release_idx=frame_idx
                    break
        frame_idx+=1
    cap.release()
    if release_idx is not None: return release_idx
    return first_ball_idx if first_ball_idx is not None else 0

def write_video(frames,out_path_mp4,fps):
    import imageio.v3 as iio
    if not frames: return None
    rgb=[cv2.cvtColor(f,cv2.COLOR_BGR2RGB) for f in frames]
    try:
        iio.imwrite(out_path_mp4,rgb,fps=max(1.0,float(fps) if fps and fps==fps else 25.0),codec='h264',quality=8)
        return out_path_mp4
    except Exception:
        h,w=frames[0].shape[:2]
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        vw=cv2.VideoWriter(out_path_mp4,fourcc,25.0,(w,h))
        for f in frames: vw.write(f)
        vw.release()
        return out_path_mp4 if os.path.exists(out_path_mp4) else None
