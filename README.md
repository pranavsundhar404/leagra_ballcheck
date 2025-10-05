
# 🏏 LEAGRA BowlCheck — YOLO Ball Release Detection Prototype

## 📖 Project Overview
**LEAGRA BowlCheck** is an **AI-powered prototype** designed to analyze bowlers’ techniques and detect **illegal bowling actions** based on **ICC’s 15° elbow rule**.  
Using **YOLOv8 Pose Estimation**, **OpenCV**, and **Streamlit**, the system identifies the **exact ball release frame** and calculates the **elbow extension angle** to determine if a delivery is legal or illegal.

---

## ✨ Features
- 🎥 Upload side-view bowling videos for AI analysis.  
- 🧠 Automatic **pose estimation** using YOLOv8 (keypoints: shoulder, elbow, wrist).  
- 🎯 Detects **true ball release frame** using AI-based motion tracking.  
- 📏 Calculates **elbow extension** to check compliance with ICC 15° rule.  
- 📊 Displays **annotated video**, **key frames**, and **verdict (Legal / Illegal)**.  
- 💬 Provides **coaching feedback** and improvement recommendations.  
- 💾 Exports **performance metrics (CSV reports)** for review.

---

## ⚙️ Installation and Setup

### Step 1: Navigate to the Project Directory
```bash
cd LEAGRA_BowlCheck_YOLO_release_ball
```

### Step 2: Create a Virtual Environment
```bash
python -m venv .venv
```

### Step 3: Activate the Environment
**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

---

## 📦 Install Dependencies
Install all required libraries using:
```bash
pip install -r requirements.txt
```

### Core Dependencies
- **Streamlit** — Web interface  
- **OpenCV** — Video frame processing  
- **Ultralytics (YOLOv8)** — Pose estimation model  
- **Torch** — Deep learning backend  
- **NumPy / Pandas / Matplotlib** — Data handling and visualization  
- **ImageIO / FFmpeg** — Video reading & writing  

---

## 🚀 Run the Application
Launch the Streamlit web app using:
```bash
streamlit run app/streamlit_app.py
```

Once the server starts, open your browser and go to:
```
http://localhost:8501
```

---

## 🧩 How It Works
1. Upload a **side-view video** of a bowler.  
2. The AI model detects **body keypoints** and tracks the **arm motion**.  
3. The system identifies the **exact frame where the ball leaves the hand**.  
4. It measures the **elbow extension angle** and checks if it exceeds 15°.  
5. Displays:
   - Key frames (arm horizontal + ball release)  
   - Annotated video with highlights  
   - Final verdict: **LEGAL** or **ILLEGAL** delivery  
6. Generates a CSV metrics report with all computed values.

---

## 🧠 Technology Stack
| Component | Technology Used |
|------------|-----------------|
| **Frontend** | Streamlit |
| **AI Model** | YOLOv8 Pose (Ultralytics) |
| **Backend** | PyTorch |
| **Video Processing** | OpenCV + ImageIO |
| **Data Analysis** | NumPy, Pandas, Matplotlib |

---

## 🔮 Future Enhancements
- Real-time live bowling detection.  
- Integration with mobile cameras.  
- 3D pose estimation for more precise motion tracking.  
- Expansion to **batting** and **fielding** AI technique analysis.  

---

## 👨‍💻 Developers
**Project Title:** LEAGRA BowlCheck — Ball Release Detection Prototype  
**Developed by:** *Pranav Sundhar & Team*  
**Organization:** LEAGRA AI Sports Tech  
**Version:** Prototype v1.0  
**License:** Educational / Research Use Only  

---

## 📫 Contact
For support or contributions, contact:  
📧 *leagra.ai.project@gmail.com*  
🌐 [LEAGRA Project Portal (Coming Soon)]()
