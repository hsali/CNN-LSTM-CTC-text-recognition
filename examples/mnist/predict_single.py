# Importing the Keras libraries and packages
from keras.models import load_model
saved_models_path = "../../saved_model"
model = load_model(saved_models_path + '/mnistCNN.h5')

from PIL import Image
import numpy as np


img = Image.open('../../images/mnist/5.png').convert("L")
img = img.resize((28,28))
im2arr = np.array(img)
im2arr = im2arr.reshape(1,28,28,1)
# Predicting the Test set results
y_pred = model.predict(im2arr)
print(y_pred)
