import numpy as np
from deepmd_jax.train import test

model_path="./model.pkl"
data_paths=[
        "CaCO3-SCAN-ML/training-set/4caco3",
        "CaCO3-SCAN-ML/training-set/ca",
        "CaCO3-SCAN-ML/training-set/caco3",
        "CaCO3-SCAN-ML/training-set/caco3-protonation",
        "CaCO3-SCAN-ML/training-set/co3",
        "CaCO3-SCAN-ML/training-set/cryst-iceIc",
        "CaCO3-SCAN-ML/training-set/cryst-iceIh",
        "CaCO3-SCAN-ML/training-set/water"
      ]

with open("pred_force.txt", "w") as f1, open("pred_energy.txt", "w") as f2:
    for data_path in data_paths:
        elemento=data_path.split("/")[-1] 
        metrics, test_results = test(model_path, data_path)

        mse_force_total = []

        for result in test_results:

            pred_force = result['predicted_force']*1000 #en meV/A
            true_force = result['true_force']*1000 #en meV/A

            Natoms = pred_force.shape[0]

            pred_energy = result['predicted_energy']*1000/Natoms #en meV/atom
            true_energy = result['true_energy']*1000/Natoms #en meV/atom

            mse_force=np.mean(np.power(pred_force.flatten()-true_force.flatten(),2))
            mse_force_total.append(mse_force)

            #Energía real y predicha en meV/átomo
            print(f"Elemento {elemento} Pred_Energy {pred_energy:.4f} True_Energy {true_energy:.4f}",  file=f2) 

            #imprimo solo una muestra de las fuerzas. medido en meV/A
            i=0
            if i % 100 == 0:
                
                filas, columnas = pred_force.shape
                for k in range(filas):
                   for j in range(columnas):
                      print(f"Elemento {elemento} Pred_Force {pred_force[k, j]:.4f} True_Force {true_force[k, j]:.4f}", file=f1)
                i += 1
            else:
                i += 1
        
        rmse_force = np.sqrt(np.mean(mse_force_total))
        print(f"RMSE_Elemento {elemento} RMSE_Force {rmse_force:.4f}", file=f1)

# mse_force=np.mean(np.power(pred_force.flatten()-true_force.flatten(),2))