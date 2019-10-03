# ImageNet Model Zoo for TensorFlow

_via Keras Applications_

_Note: this tool is largely obsolete for those using TF 2.0, which places more 
of an emphasis on Keras models._

The _TensorFlow Model Zoo_ serves to address the missing component of TensorFlow
which is present in nearly every other deep learning framework: easy access to
pre-trained ImageNet models. The ability to easily load in pre-trained models is
an enormous asset both for purposes of application deployment and for transfer
learning.

The tool is a simple wrapper around Keras's
[Applications](https://keras.io/applications/) module, which contains several
state-of-the-art classification models, along with a variety of customizable
parameters and pre-trained ImageNet weights. This code simply wraps each
function to load the model on the provided session and extracts the requisite
tensors to be returned to the user for easy use in TensorFlow.

### Usage

The API was meant to mirror the [Keras
Applications](https://keras.io/applications/) API as closely as possible, and
works very similarly, with a few extra parameters thrown in. Creating a model
returns an `X` and `Y` tensor, which correspond to the input and output tensors
of the model, as well as any other tensors indicated by the user.

```python3
from tfmodelzoo import InceptionV3
sess = tf.Session()
X, Y = InceptionV3(sess, weights='imagenet')
# X is the input tensor and Y is the output of the model (in this case, softmax)
loss = tf.losses.log_loss(labels, Y)
sess.run([loss, Y], feed_dict={X: data})
```

The graph can then be accessed (if needed) by `sess.graph`. If another tensor
(other than the input and output tensors) is needed, it can either be accessed
from the graph directly or when loading the model:

```python3
# can be accessed from the graph directly:
conv1 = sess.graph.get_tensor_by_name('conv1tensorname:0')
# or through the `include_tensors` convenience parameter when loading the model:
X, Y, conv1 = InceptionV3(sess, include_tensors=['conv1tensorname:0'])
```

It also imports the [Keras ImageNet
utils](https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py)
for you, making it easy to preprocess
inputs and decode your predictions back to their text labels:

```python3
from tfmodelzoo import preprocess_input, decode_predictions
data = preprocess_input(images)
predictions = sess.run(Y, feed_dict={X: data})
pred_labels = decode_predictions(predictions, top=3)
#[[('n02088364', 'beagle', 0.8906793),
#  ('n02092339', 'Weimaraner', 0.06135427),
#  ('n02108000', 'EntleBucher', 0.012612144)]]
```

See the [demo notebook](demo/demo.ipynb) for an example. Beyond that, reference
the [Keras Documentation](https://keras.io/applications/) for information on the
arguments that may be passed for each model.

_Note: Passing a Keras `Input`, as indicated in the documentation, works, but as
of now there is no way to do this with pure TensorFlow_.

### Available Models:

Users may refer to the [Keras Documentation](https://keras.io/applications/) for
detailed information about the models available. That said, the models currently
supported are:

- Xception
- VGG16
- VGG19
- ResNet50
- InceptionV3
- InceptionResNetV2
- MobileNet
- DenseNet121
- DenseNet169
- DenseNet201

### Setup:
With pip, simply run:
```
pip install -e git+https://github.com/joeddav/tensorflow-modelzoo.git#egg=tfmodelzoo
```
