def generate_report(prediction, confidence):
    if prediction == "healthy":
        return "No abnormality detected. Retina appears healthy."
    elif prediction == "dr":
        return "Signs of Diabetic Retinopathy detected. Consultation advised."
    elif prediction == "glaucoma":
        return "Possible Glaucoma detected. Further tests recommended."
    elif prediction == "amd":
        return "Macular degeneration detected. Immediate attention needed."
