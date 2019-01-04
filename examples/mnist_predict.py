from keras.models import model_from_json

saved_model = "../saved_model/mnist"

json_file = open(saved_model+ "model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(saved_model + "/model.h5")

print("loaded model from disk")

loaded_model.compile(loss="binary_crossentropy", optimizer = 'rmsprop', metrics = ['accuracy'])
