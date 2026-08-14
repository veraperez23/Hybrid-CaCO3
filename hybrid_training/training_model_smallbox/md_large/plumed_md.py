import plumed
import numpy as np
from deepmd_jax.md import DPJaxCalculator
from ase import Atoms
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units
from ase.io import read, write
from ase.md import MDLogger
from ase.calculators.plumed import Plumed

model_path="../model.pkl"
atoms = read("caco3-hybrid-final.lammps-data")
symbols = atoms.get_chemical_symbols()
type_map = {0: "Ca", 1: "C", 2: "O", 3: "H"}
symbol_to_type = {v: k for k, v in type_map.items()}
type_idx = np.array([symbol_to_type[s] for s in symbols])

calc_dp = DPJaxCalculator(model_path=model_path, type_idx=type_idx)

with open("plumed.dat") as f:
    setup = [line.strip() for line in f]

# Initialize velocities.
T_init = 330  # Desfase de 30K en el resultado de la temperatura
MaxwellBoltzmannDistribution(atoms, temperature_K=T_init)
kB = 0.00008617333262
timestep = 0.5 * units.fs

atoms.calc = Plumed(calc=calc_dp, input=setup, timestep=timestep, atoms=atoms, kT=kB*T_init)

# Some parameters
ttime = 100 * units.fs
ptime = 1000 * units.fs
bulk_modulus = 10.0
pfactor = (ptime**2) * bulk_modulus * units.GPa

# Define dump coordinates function
def write_frame():
        dyn.atoms.write('md_hybrid.xyz', append=True)

# Function to remove COM velocity
def remove_com():
    Stationary(atoms)

print("Starting MD")
# Set up the NPT ensemble.
dyn = NPT(atoms, timestep, temperature_K=T_init, externalstress=0.0, ttime=ttime, pfactor=pfactor, mask=np.array([[1,0,0],[0,1,0],[0,0,1]]))
dyn.set_fraction_traceless(0) #mantiene la dimensión de la caja constante

dyn.attach(write_frame, interval=2000)
logger = MDLogger(dyn, atoms, 'md_hybrid.log', stress=True, peratom=False, mode="a")
dyn.attach(logger, interval=200)
dyn.attach(remove_com, interval=200000)
#dyn.zero_center_of_mass_momentum()

n_steps = 4000000 # Number of steps to run
dyn.run(n_steps)
print("MD done!")

write("caco3-hybrid-final-final.lammps-data",atoms)
