def evaluate_alerts(face_quality_results):
    """
    Check if a face quality status warrants a security alert warning.
    """
    status = face_quality_results.get('status', 'Normal')
    is_alert = False
    alert_message = ""
    
    if status in ["Face Too Small", "Face Occluded", "Low Confidence"]:
        is_alert = True
        alert_message = f"Warning: {status}"
    elif status in ["Head Down", "Head Up", "Head Left", "Head Right"]:
        is_alert = True
        alert_message = f"Warning: {status}"
        
    return is_alert, alert_message
