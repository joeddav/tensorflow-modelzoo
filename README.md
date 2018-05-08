# ImageNet Model Zoo for TensorFlow
_via Keras Applications_

The _TensorFlow Model Zoo_ serves to address the missing component of TensorFlow which is present in nearly every other deep learning framework: easy access to pre-trained ImageNet models. The ability to easily load in pre-trained models is an enormous asset both for purposes of application deployment and for transfer learning.

The tool is a simple wrapper around Keras's [Applications](https://keras.io/applications/) module, which contains several state-of-the-art classification models, along with a variety of customizable parameters and pre-trained ImageNet weights. This code simply wraps each function and extracts the underlying session (and graph), and the requisite tensors to be returned to the user for easy use in TensorFlow.

### Usage

The API was meant to mirror the [Keras Applications](https://keras.io/applications/) API as closely as possible, and works very similarly, with a few extra parameters thrown in. Creating a model returns the TensorFlow `Session` on which the graph is built, as well an `X` and `Y` tensor, which correspond to the input and output tensors of the model.

```python3
from tfmodelzoo import InceptionV3
sess, X, Y = InceptionV3(include_top=True, weights='imagenet')
# X is the input tensor and Y is the last operation of the model (in this case, softmax)
loss = tf.losses.log_loss(labels, Y)
sess.run([loss, Y], feed_dict={X: data})
```

The graph can then be accessed (if needed) by `sess.graph`. If another tensor (other than the input and output tensors) is needed, it can be accessed from the graph directly or when loading the model:

```python3
conv1 = sess.graph.get_tensor_by_name('conv1tensorname:0') # access from graph directly OR
sess, X, Y, conv1 = InceptionV3(include_tensors=['conv1tensorname:0']) # access via convenience parameter
```

By default, the model is loaded onto a new session which is returned to the user. A custom session can also be passed to the API, however:

```python3
sess = tf.Session()
...
_, X, Y = InceptionV3(sess=sess)
```

Beyond that, see the [Keras Documentation](https://keras.io/applications/) for information on the arguments that may be passed for each model.

_Note: Passing a Keras `Input`, as indicated in the documentation, works, but as of now there is no way to do this with pure TensorFlow_.

### Setup:
With pip, simply run:
```
pip install -e git+https://github.com/joeddav/tensorflow-modelzoo.git#egg=tfmodelzoo
```
