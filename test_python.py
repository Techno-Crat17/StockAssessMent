from keras.models import load_model

model = load_model("keras_model.keras", compile=False)

print("Model loaded successfully")