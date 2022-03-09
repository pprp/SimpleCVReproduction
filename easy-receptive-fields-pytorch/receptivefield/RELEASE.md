# Release 0.5.0

## Major Features And Improvements
* Migration to Tensorflow (using Keras API) >2.0 and Pytorch >1.3
* Old API still can be accessed by using older version of the library (0.4.0)

# Release 0.4.0

## Major Features And Improvements
* Added initial Pytorch API with `PytorchReceptiveField`.

# Release 0.3.0

## Major Features And Improvements
* Update Keras and Tensorflow api for multiple feature maps calculation
* Change of api in compute function: removed `output_layer: str` 
and replaced with `output_layers: List[str]`, which can accept a list 
of names for feature maps for which we want to compute receptive fields.
* Change in `plot_gradient_at`, added new parameter which control for 
which feature map gradient is plotted. Same for `plot_rf_grid` function.
* New functions for plotting: `plot_rf_grids` and `plot_gradients_at`.
* Improved tensorflow API, now one does have to initialize model with 
InteractiveSession, this is handled by the `TFReceptiveField`.
* A new API for tensorflow: `TFFeatureMapsReceptiveField`.