from keras.models import load_model

model = load_model("keras_model.keras", compile=False)

model.save("keras_model.h5")

print("Model converted successfully")