def run_diagnosis(model, image, conf_threshold=0.25):
    return model.predict(image, conf=conf_threshold)