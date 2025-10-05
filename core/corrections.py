def generate_corrections(summary, verdict_text=''):
    tips=[]
    if 'ILLEGAL' in verdict_text or 'SUSPECT' in verdict_text:
        tips.append('⚠️ Potential illegal elbow extension at release (>15°). Keep the forearm–upper arm angle more open through release.')
    if summary.get('max_trunk_lean_deg',0)>25:
        tips.append('Reduce trunk side-lean; stay taller at the crease for balance and repeatability.')
    mean_sh=summary.get('mean_shoulder_tilt_deg',0)
    if abs(mean_sh)>15:
        tips.append('Shoulder line is tilted; align your shoulders towards target to improve control.')
    if not tips:
        tips.append('Looks solid ✅. Maintain a smooth run-up, strong front-arm pull, and stable head position.')
    tips.append('Drills: mirror shadow bowling, single-arm wall drill, slow-mo walk-ins focusing on straight arm path.')
    return "\n".join(f"- {t}" for t in tips)
