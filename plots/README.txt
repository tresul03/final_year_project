plots: contains plots produced by the scripts in src.

iterative_training_vs_fixed_dataset:
    Contains plots of target function and model's fit to said function.
    Model is iteratively retrained by training it on new data each iteration.
    How the model is retrained differs. The model either:
        Receives randomly sampled data (random retraining), or
        Receives data based on output regions of greatest standard deviation (selectve retraining)
    
    fixed_n_plots.pdf --> no iterative retraining. Model is trained once on n number of data points, before fitting to target function.
    random_n_plots.pdf --> random retraining
    selective_n_plots.pdf --> selective retraining

model_comparator:
    2D plot of residuals and standard deviation of outputs as functions of inputs.
    unordered plot succeeds ordered plot.

mse_vs_size:
    table of model mse against dataset size, for both iterative training methods.
