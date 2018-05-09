# Let's look at how the API is used by importing a simple pretrained imagenet
# model and testing it out on a picture of a dog. We start with our imports, and
# can import our ResNet50 model-getter and decode_predictions tool for
# retrieving labels

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tfmodelzoo import ResNet50, decode_predictions

dog_img = plt.imread('dog.jpg')

# Create a session and pass it to our model getter. This will download the
# weights and return two tensors: the input and the output of the model

sess = tf.Session()
data, softmax = ResNet50(sess, weights='imagenet')

# We can now go ahead and use these models in TensorFlow, adding more operations
# or simply running them. I'll put my dog through the model and see how it does:

predictions = sess.run(softmax, {data: np.array([dog_img])})

# Now we can decode our predictions:

print('Top 10 Predictions w/ Confidence:')
for i, tup in enumerate(decode_predictions(predictions, top=10)[0]):
    print("{}\t— {}".format(tup[2], tup[1]))

# Top 10 Predictions w/ Confidence:
# 0.8906792998313904    — beagle
# 0.061354268342256546  — Weimaraner
# 0.012612144462764263  — EntleBucher
# 0.010248360224068165  — Walker_hound
# 0.006116565316915512  — English_foxhound
# 0.004969390109181404  — bluetick
# 0.0017149088671430945 — redbone
# 0.001614996581338346  — Labrador_retriever
# 0.0015570599352940917 — basset
# 0.0012122730258852243 — dalmatian

# The above should be the output when the model is run!
