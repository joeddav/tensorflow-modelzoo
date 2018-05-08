import tensorflow as tf

def _get_model(sess, model, include_tensors, **kwargs):
    # preserve previous keras session and create a new one
    old_sess = tf.keras.backend.get_session()
    if sess is None:
        tf.keras.backend.clear_session()
    else:
        tf.keras.backend.set_session(sess)

    # load Keras model via applications module
    mod = model(**kwargs)

    # get input tensor for feeddict needs and get output tensor
    X = mod.get_input_at(0)
    Y = mod.get_output_at(0)

    # get additional tensors from graph as requested by user
    tensors = []
    sess = tf.keras.backend.get_session()
    for name in include_tensors:
        tensors.append(sess.graph.get_tensor_by_name(name))

    tf.keras.backend.set_session(old_sess)
    return (sess, X, Y, *tensors)


def DenseNet121(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#densenet for more information and
    documentation.

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#densenet. The standard are
                consist of the following parameters, but certain models have
                additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.DenseNet121,
                    include_tensors, **kwargs)


def DenseNet169(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#densenet for more information and
    documentation.

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#densenet. The standard are
                consist of the following parameters, but certain models have
                additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.DenseNet169,
                    include_tensors, **kwargs)


def DenseNet201(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#densenet for more information and
    documentation.

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#densenet. The standard are
                consist of the following parameters, but certain models have
                additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.DenseNet201,
                    include_tensors, **kwargs)


def InceptionResNetV2(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#inceptionresnetv2 for more information and
    documentation.

    Min image size: 139x139x3

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#inceptionresnetv2. The standard
                are consist of the following parameters, but certain models
                have additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.InceptionResNetV2,
                    include_tensors, **kwargs)


def InceptionV3(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#inceptionv3 for more information and
    documentation.

    Min image size: 139x139x3

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#inceptionv3. The standard are
                consist of the following parameters, but certain models have
                additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.InceptionV3,
                    include_tensors, **kwargs)


def MobileNet(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#mobilenet for more information and
    documentation.

    Min image size: 32x32x3

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#mobilenet. The standard are
                consist of the following parameters, but certain models have
                additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.MobileNet,
                    include_tensors, **kwargs)


def NASNetLarge(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#nasnet for more information and
    documentation.

    Min image size: 32x32x3

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#nasnet. The standard are
                consist of the following parameters, but certain models have
                additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.NASNetLarge,
                    include_tensors, **kwargs)


def NASNetMobile(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#nasnet for more information and
    documentation.

    Min image size: 32x32x3

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#nasnet. The standard are
                consist of the following parameters, but certain models have
                additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.NASNetMobile,
                    include_tensors, **kwargs)


def ResNet50(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#resnet50 for more information and
    documentation.

    Min image size: 197x197x3

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#resnet50. The standard are
                consist of the following parameters, but certain models have
                additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.ResNet50,
                    include_tensors, **kwargs)


def VGG16(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#vgg16 for more information and
    documentation.

    Min image size: 48x48

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#vgg16. The standard are
                consist of the following parameters, but certain models have
                additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.VGG16,
                    include_tensors, **kwargs)


def VGG19(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#vgg19 for more information and
    documentation.

    Min image size: 48x48

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#vgg19. The standard are
                consist of the following parameters, but certain models have
                additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.VGG19,
                    include_tensors, **kwargs)


def Xception(sess=None, include_tensors=[], **kwargs):
    """DenseNet pre-trained on ImageNet, returning the session and the input and
    output tensors. Graph accessible via `sess.graph`. See
    https://keras.io/applications/#xception for more information and
    documentation.

    Min image size: 71x71x3

    # Arguments:
        sess (Session): optional TensorFlow session to load the model and
                        weights on. If not passed, a new session will be
                        created. In either case, the session is the first value
                        returned from the function.
        include_tensors: a convenience parameter to return additional tensors
                        from the graph in addition to the input and output
                        layers. Can also be done by calling
                        `sess.graph.get_tensor_by_name(name)`.
        **kwargs: arguments to be passed to the Keras application. See
                https://keras.io/applications/#xception. The standard are
                consist of the following parameters, but certain models have
                additional options as well. Note also that the `input`
                parameter in the Keras documentation is not currently
                supported.
            include_top (bool): whether to include the fully connected layers of
                                the network at the end
            weights: 'imagenet' to include pre-trained ImageNet weights, None to
                    exclude pretrained weights, or the path to the weights file
                    to be loaded.
            input_shape (tuple): shape of input image
            pooling: optional pooling mode for feature extraction when
                    include_top is False. `None` means that the output of the
                    model will be the 4D tensor output of the last
                    convolutional layer. `avg` means that global average
                    pooling will be applied to the output of the last
                    convolutional layer, and thus the output of the model will
                    be a 2D tensor. `max` means that global max pooling will be
                    applied.
            classes: optional number of classes to classify images into, only to
                    be specified if `include_top` is `True`, and if no weights
                    argument is specified.

    # Returns:
        session: the new session instance on which the graph is built
        X: the input tensor of the graph
        Y: the output tensor of the graph (softmax or pooling layer)
        *tensors: any other tensors requested by the user via the
                `include_tensors` parameter
    """
    return _get_model(sess, tf.keras.applications.Xception,
                    include_tensors, **kwargs)
