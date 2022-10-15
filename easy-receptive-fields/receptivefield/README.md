<img src="img/demo_readme.jpg" width="800">

# receptivefield

Gradient based receptive field estimation for Convolutional 
Neural Networks. **receptivefield** uses backpropagation of 
the gradients from output of selected feature maps to the input image in order to
estimate the size (width, height), stride and offset of resulting
receptive field. Numerical estimation of receptive field can be 
useful when dealing with more complicated neural networks like
ResNet, Inception (see notebooks) where analytical approach of 
computing receptive fields cannot be used.

[![Build Status](https://travis-ci.org/fornaxai/receptivefield.svg?branch=master)](https://travis-ci.org/fornaxai/receptivefield)

# Installation & supported APIs

* Requires: python (in version >= 3.6), numpy, matplotlib, pillow (check requirements.txt)
* `pip install receptivefield>=0.5.0`
    * tensorflow.keras>=2.0
    * pytorch>=1.3.1

* `pip install receptivefield==0.4.0`
    * keras==2.1.6 
    * tensorflow==1.8.0
    * pytorch>=0.4.0


# Some remarks

* In order to get better results or avoid possible NaNs in the 
estimated receptive field parameters, it is suggested to 
use `Linear` (instead `Relu`) activation and `AvgPool2D` instead of `MaxPool2D`.
This improves gradient flow in the network and hence better signal
in the input image. Note, that this is required only for RF estimation.

* Additionally, one may even initialize network with constant 
positive values in all weights (positive if max pooling is used)
and set biases to zero. In the case of Keras API this can be obtained by setting `init_weight=True` 
in the `KerasReceptiveField(init_weight=True)` constructor.

# Limitations

* Numerical approach cannot be used when RF is larger that input image, however 
one may try to increase the input image size, since RF parameters depend on the architecture not 
input image size.

# Supported APIs

Currently only Keras, Tensorflow and Pytorch API are supported. However, it should be
possible to extend **receptivefield** functionality by deriving
abstract class **ReceptiveField** in the base.py file. 

* Keras: `KerasReceptiveField`, example usage in *notebooks/keras_api.ipynb*
* Tensorflow: `TFReceptiveField` or `TFFeatureMapsReceptiveField`, example usage in *notebooks/tensorflow_api.ipynb*
* Pytorch: `PytorchReceptiveField`, example usage in *notebooks/minimal_example_with_pytorch_API.ipynb*

# How does it work?

This is description of a general approach which is almost the same for other
APIs.

1. Define build_function which returns model (here Keras model)

    ```python
    def model_build_func(input_shape=[224, 224, 3]):
        ...
        return Model(input, output)
    ```

2. Compute receptive field parameters with `KerasReceptiveField`

    ```python
    from receptivefield.keras import KerasReceptiveField
    rf_params = KerasReceptiveField(model_build_func).compute(
        input_shape=[224, 224, 3], # this will be passed to model_build_func
        input_layer='input_image', # must exist - usually input image layer
        output_layers=['feature_map'] # for example last conv layer
    )
    ```
    
3. The `rf_params` is a list of `FeatureMapDescription`. Here we selected 
only single feature map `output_layers=['feature_map']` and the `rf_params` will 
contain
  
    ```python
    rf_params = [
       FeatureMapDescription(
           size=Size(w=60, h=60), 
           rf=ReceptiveFieldDescription(
               offset=(2.5, 2.5), 
               stride=(1.0, 1.0), 
               size=Size(w=9, h=9))
           )
    ]
    ```
    Explanation of `FeatureMapDescription` fields:
    * `size` - defined the spatial dimensions of the feature map i.e. the width 
        and height of the feature map grid.
    * rf is an instance of `ReceptiveFieldDescription`:
        * `offset` - defines location of the first left-top anchor in the 
        image coordinates (defined in pixels). 
        * `stride` - defines how much RF of the network moves w.r.t unit displacement 
        in the feature_map tensor. 
        * `size` - defines the effective area in the input image which one point 
        in the feature_map tensor is seeing.

# Keras minimal - copy/paste example

* Python code:

    ```python
    from keras.layers import Conv2D, Input, AvgPool2D
    from keras.models import Model
    from receptivefield.image import get_default_image
    from receptivefield.keras import KerasReceptiveField

    # define model function
    def model_build_func(input_shape):
        act = 'linear' # see Remarks
        inp = Input(shape=input_shape, name='input_image')
        x = Conv2D(32, (7, 7), activation=act)(inp)
        x = Conv2D(32, (5, 5), activation=act)(x)
        x = AvgPool2D()(x)
        x = Conv2D(64, (5, 5), activation=act, name='feature_grid')(x)
        x = AvgPool2D()(x)
        model = Model(inp, x)
        return model

    shape = [64, 64, 3]
    # compute receptive field
    rf = KerasReceptiveField(model_build_func, init_weights=True)
    rf_params = rf.compute(shape, 'input_image', ['feature_grid'])
    # debug receptive field
    rf.plot_rf_grids(get_default_image(shape, name='doge'))
    ```
* Logger output + example RF grid
    ```text
      [ INFO][keras.py]::Feature maps shape: [[None, 23, 23, 1]]
      [ INFO][keras.py]::Input shape       : (None, 64, 64, 3)
      [ INFO][base.py]::Estimated receptive field for feature map [0]: ReceptiveFieldDescription(offset=(10.0, 10.0), stride=(2.0, 2.0), size=Size(w=20, h=20))
    ```
    
    <img src="img/demo_minimal.jpg" width="400">


# Pytorch minimal - copy/paste example

* Python code:

    ```python
    import torch.nn as nn
    from receptivefield.pytorch import PytorchReceptiveField
    from receptivefield.image import get_default_image
    
    class Linear(nn.Module):
        """An identity activation function"""
        def forward(self, x):
            return x
    # define some example feature extractor, here we compute RFs for two 
    # feature maps
    class SimpleVGG(nn.Module):
        def __init__(self, disable_activations: bool = False):
            """disable_activations: whether to generate network with Relus or not."""
            super(SimpleVGG, self).__init__()
            self.features = self._make_layers(disable_activations)
    
        def forward(self, x):
            # index of layers with feature maps
            select = [8, 13]
            # self.feature_maps is a list of Tensors, PytorchReceptiveField looks for 
            # this parameter and compute receptive fields for all Tensors inside it.
            self.feature_maps = []
            for l, layer in enumerate(self.features):
                x = layer(x)
                if l in select:
                    self.feature_maps.append(x)
            return x
    
        def _make_layers(self, disable_activations: bool):
            activation = lambda: Linear() if disable_activations else nn.ReLU()
            layers = [
                nn.Conv2d(3, 64, kernel_size=3),
                activation(),
                nn.Conv2d(64, 64, kernel_size=3),
                activation(),
                
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3),
                activation(),
                nn.Conv2d(128, 128, kernel_size=3),
                activation(), # 8
                
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3),
                activation(),
                nn.Conv2d(256, 256, kernel_size=3),
                activation(), # 13
            ]        
            return nn.Sequential(*layers)    
    
    # define model functions
    def model_fn() -> nn.Module:
        model = SimpleVGG(disable_activations=True)
        model.eval()
        return model

    input_shape = [96, 96, 3]
    rf = PytorchReceptiveField(model_fn)
    rf_params = rf.compute(input_shape = input_shape)
    # plot receptive fields
    rf.plot_rf_grids(
        custom_image=get_default_image(input_shape, name='cat'), 
        figsize=(20, 12), 
        layout=(1, 2))
    ```
    
* Logger output
    ```text
    [ INFO][pytorch.py]::Feature maps shape: [GridShape(n=1, w=42, h=42, c=128), GridShape(n=1, w=17, h=17, c=256)]
    [ INFO][pytorch.py]::Input shape       : GridShape(n=1, w=96, h=96, c=3)
    [ INFO][base.py]::Estimated receptive field for feature map [0]: ReceptiveFieldDescription(offset=(7.0, 7.0), stride=(2.0, 2.0), size=Size(w=14, h=14))
    [ INFO][base.py]::Estimated receptive field for feature map [1]: ReceptiveFieldDescription(offset=(16.0, 16.0), stride=(4.0, 4.0), size=Size(w=32, h=32))
    ```
    
# Keras more detailed example

Here we show, how to estimate effective receptive field of any Keras model.

* Create model build_function which returns model. This function
should accept one parameter `input_shape`.
    
    ```python
    from keras.layers import Conv2D, Input
    from keras.layers import AvgPool2D
    from keras.models import Model
    
    def model_build_func(input_shape):
        activation = 'linear'
        padding='valid'
        
        inp = Input(shape=input_shape, name='input_image')
        x = Conv2D(32, (5, 5), padding=padding, activation=activation)(inp)
        x = Conv2D(32, (3, 3), padding=padding, activation=activation)(x)
        x = AvgPool2D()(x)
        x = Conv2D(64, (3, 3), activation=activation, padding=padding)(x)
        x = Conv2D(64, (3, 3), activation=activation, padding=padding)(x)
        x = AvgPool2D()(x)
        x = Conv2D(128, (3, 3), activation=activation, padding=padding)(x)
        x = Conv2D(128, (3, 3), activation=activation, padding=padding, name='feature_grid')(x)
    
        model = Model(inp, x)
        return model
    ```

* Check if model is building properly:
    ```python
    model = model_build_func(input_shape=(96, 96, 3))
    model.summary()
    ```
    
    ```text
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_image (InputLayer)     (None, 96, 96, 3)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 92, 92, 32)        2432      
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 90, 90, 32)        9248      
    _________________________________________________________________
    average_pooling2d_1 (Average (None, 45, 45, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 43, 43, 64)        18496     
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 41, 41, 64)        36928     
    _________________________________________________________________
    average_pooling2d_2 (Average (None, 20, 20, 64)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 18, 18, 128)       73856     
    _________________________________________________________________
    feature_grid (Conv2D)        (None, 16, 16, 128)       147584    
    =================================================================
    Total params: 288,544
    Trainable params: 288,544
    Non-trainable params: 0
    ```

* This step is not required but it is useful to plot results in the
example image. For instance, you would like to see what is the size
of network receptive field in comparision to some objects you
wish detect (or localize) by this network.
    
    ```python
    from receptivefield.image import get_default_image
    import matplotlib.pyplot as plt
    # Load sample image of `Lena`.
    image = get_default_image(shape=(32, 32), tile_factor=1)
    plt.imshow(image)
    ```
    
    <img src="img/demo_keras_lena.jpg" width="512">

* Compute receptive field of the network by calling `rf.compute`

    ```python
    from receptivefield.keras import KerasReceptiveField
    
    rf = KerasReceptiveField(model_build_func, init_weights=False)
    
    rf_params = rf.compute(
        input_shape=image.shape, 
        input_layer='input_image', 
        output_layers=['feature_grid']
    )
    print(rf_params)
    ```

* The resulting receptive field is:

    ```text
    [FeatureMapDescription(
       size=Size(w=16, h=16), 
       rf=ReceptiveFieldDescription(
           offset=(17, 17), 
           stride=(4.0, 4.0), 
           size=Size(w=34, h=34))
    )]
    ```

* Input shape: `rf.input_shape==ImageShape(w=96, h=96, c=3)`
* Output feature map spatial dimensions: `rf.output_shapes==[Size(w=16, h=16)]`.
* You may want to see how gradients backpropagate to the input image. Here
`point=(8, 8)` refers to the (W, H) position of the source signal
from the output grid. Here we select the first feature map with `fm_id=0`.
    
    ```python
    
    rf.plot_gradient_at(fm_id=0, point=(8, 8), image=None, figsize=(7, 7))
    ```
    
    <img src="img/demo_keras_response.jpg" width="512">

* Or even plot whole receptive field grid:

    ```python
    rf.plot_rf_grids(custom_image=image, figsize=(6, 6))
    ```
    
    <img src="img/demo_keras_rf_debug.jpg" width="512">

* In the above, the red rectangle corresponds to the area which top-left
grid point is seeing in the input image. Blue rectangle corresponds
to the central grid point, green to the bottom-right point. Green dots
show the position of the centers of the grid anchors in the source
image. 

* For more plotting functions check the implementation of the base class 
`ReceptiveField` defined in `base.py` file.
