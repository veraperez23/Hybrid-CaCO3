from deepmd_jax.train import train

# training an energy-force model
train(
      model_type='energy',              # Model type
      rcut=6.0,                              # Cutoff radius
      save_path='model.pkl',  # Path to save the trained model
      train_data_path=[
        "CaCO3-SCAN-ML/training-set/4caco3",
        "CaCO3-SCAN-ML/training-set/ca",
        "CaCO3-SCAN-ML/training-set/caco3",
        "CaCO3-SCAN-ML/training-set/caco3-protonation",
        "CaCO3-SCAN-ML/training-set/co3",
        "CaCO3-SCAN-ML/training-set/cryst-iceIc",
        "CaCO3-SCAN-ML/training-set/cryst-iceIh",
        "CaCO3-SCAN-ML/training-set/water"
      ], # Path (or a list of paths) to the training dataset
      step=2000000,                          # Number of training steps
      print_every = 100,
)