from typing import Tuple, Callable, Any, List
import tensorflow as tf
import numpy as np


import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Activation
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers


from receptivefield.base import ReceptiveField
from receptivefield.common import scaled_constant
from receptivefield.logging import get_logger
from receptivefield.types import ImageShape, GridPoint, GridShape, FeatureMapDescription


_logger = get_logger()


def _check_activation(layer: Layer):
    if issubclass(type(layer), Layer):
        layer_act = layer.activation.__name__
    else:
        layer_act = None

    if layer_act != "linear":
        _logger.warning(
            f"Layer {layer.name} activation should be linear " f"but is: {layer_act}"
        )


def safe_init_conv2d(layer: layers.Conv2D):

    weights = layer.get_weights()
    w_kernel = scaled_constant(1, weights[0].shape)

    _logger.info(
        f"Setting weights for layer `{layer.name}` :: "
        f"{layer.__class__.__name__}{w_kernel.shape}"
    )

    w = [w_kernel]
    if len(weights) > 1:
        w_bias = np.zeros_like(weights[1])
        w.append(w_bias)
    layer.set_weights(w)
    _check_activation(layer)


def setup_model_weights(model: Model) -> None:
    """
    Set all weights to be a constant values. Biases are set to zero.
    Only Conv2D are supported.

    :param model: a Keras model
    """

    if isinstance(model, Model):
        _logger.info(f"Running `setup_model_weights` on Model: {type(model)}")
        _layers = model.layers
    elif isinstance(model, layers.Layer):
        _logger.info(f"Running `setup_model_weights` on Layer: {type(model)}")
        _layers = model._layers

    for layer in _layers:
        # check layer type
        if type(layer) == MaxPool2D:
            _logger.warning(
                f"MaxPool2D detected: {layer.name}. Replace it with"
                f" AvgPool2D in order to obtain better receptive "
                f"field mapping estimation"
            )

        if type(layer) in [layers.AvgPool2D, layers.InputLayer]:
            continue

        # set weights
        if type(layer) == Conv2D:
            safe_init_conv2d(layer)
        elif type(layer) == Activation:
            _check_activation(layer)
        elif isinstance(layer, layers.Layer):
            setup_model_weights(layer)
        else:
            _logger.warning(
                f"Setting weights for layer {type(layer)} is not supported."
            )


def _define_receptive_field_func(
    model: Model, input_layer: str, feature_map_layers: List[str]
):

    output_shapes = []
    input_shape = model.get_layer(input_layer).output_shape[0]

    for fm in feature_map_layers:
        output_shape = model.get_layer(fm).output_shape
        output_shape = [*output_shape[:-1], 1]
        output_shapes.append(output_shape)

    def gradients_fn(receptive_field_masks, model_input, **kwargs):

        outputs = []
        for k, fm in enumerate(feature_map_layers):
            outputs.append(model.get_layer(fm).output)

        fm_model = Model(model.get_layer(input_layer).input, outputs)
        grads = []
        for fm_id, receptive_field_mask in enumerate(receptive_field_masks):
            with tf.GradientTape() as tape:
                x = fm_model(model_input)[fm_id]
                x = Lambda(lambda _x: K.mean(_x, -1, keepdims=True))(x)
                fake_loss = x * receptive_field_mask
                fake_loss = K.mean(fake_loss)
                grad = tape.gradient(fake_loss, model_input)
                grads.append(grad)

        return grads

    _logger.info(f"Feature maps shape: {output_shapes}")
    _logger.info(f"Input shape       : {input_shape}")

    return gradients_fn, input_shape, output_shapes


class KerasReceptiveField(ReceptiveField):
    def __init__(
        self, model_func: Callable[[ImageShape], Any], init_weights: bool = False
    ):
        """
        Build Keras receptive field estimator.

        :param model_func: model creation function
        :param init_weights: if True all conv2d weights are overwritten
        by constant value.
        """
        super().__init__(model_func)
        self.init_weights = init_weights

    def _prepare_gradient_func(
        self, input_shape: ImageShape, input_layer: str, output_layers: List[str]
    ) -> Tuple[Callable, GridShape, List[GridShape]]:
        """
        Computes gradient function and additional parameters. Note
        that the receptive field parameters like stride or size, do not
        depend on input image shape. However, if the RF of original network
        is bigger than input_shape this method will fail. Hence it is
        recommended to increase the input shape.

        :param input_shape: shape of the input image. Used in @model_func.
        :param input_layer: name of the input layer.
        :param output_layers: a list of names of the target feature map layers.

        :returns
            gradient_function: a function which returns gradient w.r.t. to
                the input image
            input_shape: a shape of the input image tensor
            output_shape: a shapes of the output feature map tensors
        """
        model = self._model_func(ImageShape(*input_shape))
        if self.init_weights:
            setup_model_weights(model)

        gradient_function, input_shape, output_shapes = _define_receptive_field_func(
            model, input_layer, output_layers
        )

        return (
            gradient_function,
            GridShape(*input_shape),
            [GridShape(*output_shape) for output_shape in output_shapes],
        )

    def _get_gradient_from_grid_points(
        self, points: List[GridPoint], intensity: float = 1.0
    ) -> List[np.ndarray]:
        """
        Computes gradient at input_layer (image_layer) generated by
        point-like perturbation at output grid location given by
        @point coordinates.

        :param points: source coordinate of the backpropagated gradient for each
            feature map.
        :param intensity: scale of the gradient, default = 1
        :return gradient maps for each feature map
        """
        input_shape = self._input_shape.replace(n=1)
        output_feature_maps = []
        for fm in range(self.num_feature_maps):
            output_shape = self._output_shapes[fm].replace(n=1)
            output_feature_map = np.zeros(shape=output_shape)
            output_feature_map[:, points[fm].y, points[fm].x, 0] = intensity
            output_feature_maps.append(output_feature_map)

        receptive_field_grads = self._gradient_function(
            output_feature_maps,
            tf.Variable(np.zeros(shape=input_shape).astype(np.float32)),
        )

        return receptive_field_grads

    def compute(
        self, input_shape: ImageShape, input_layer: str, output_layers: List[str]
    ) -> List[FeatureMapDescription]:

        """
        Compute ReceptiveFieldDescription of given model for image of
        shape input_shape [H, W, C]. If receptive field of the network
        is bigger thant input_shape this method will raise exception.
        In order to solve with problem try to increase input_shape.

        :param input_shape: shape of the input image e.g. (224, 224, 3)
        :param input_layer: name of the input layer
        :param output_layers: a list of names of the target feature map layers.

        :return a list of estimated FeatureMapDescription for each feature
            map.
        """

        return super().compute(
            input_shape=input_shape,
            input_layer=input_layer,
            output_layers=output_layers,
        )
