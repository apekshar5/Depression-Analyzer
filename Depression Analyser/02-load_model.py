from keras.models import load_model

# load model
model = load_model('model.h5')

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('image1.jpg', target_size = (28,28))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

predicted_class_indices=np.argmax(result,axis=1)

labels = {'depressed': 0, 'not depressed': 1}

labels = dict((v,k) for k,v in labels.items())
prediction = [labels[k] for k in predicted_class_indices]
print(prediction[0])
