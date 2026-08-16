import jax
from deepmd_jax.train import train


# training an energy-force model
train(
      model_type='energy',              # Model type
      init_model='/scratch/vperez/ForVera/ExampleTrainDeePMD_Jax_2/training_model/model.pkl',
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
      step=100000,                          # Number of training steps
      print_every = 10, #puede que sea muy alto, a lo mejor hay que bajarlo 
      hybrid=True,
      obs_train_data_path=[
          '../../Ab_initio_training/md_small_2ns/dataset',
      ],
      obs_temperature = [330], # Temperature in Kelvin
      obs_target = [-18.*0.010364269666250866],
      obs_batch_size = 150,
      obs_s_pref = 1,
      obs_l_pref = 100,
      obs_r1 = 0.9,
      obs_r2 = 1.0
)