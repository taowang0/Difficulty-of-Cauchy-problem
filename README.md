## Burgers' equation simulation 1D

#### Source and citation: 
[https://github.com/sachabinder/Burgers\_equation\_simulation/tree/main](https://github.com/sachabinder/Burgers_equation_simulation/tree/main)

> Sacha BINDER. Étude de l’observation et de la modélisation des ondes de surface en eau peu profonde. Physique-Informatique.TIPE session 2021.

Run solver:

```
python Burgers_solver_SP.py
```

Solution is saved in `data/U.npy`. A 3D plot of the solution is saved in `figures/Burgers_solution.pdf`.

## Neural network fitting (for MSE loss)

Reproduce Figure 7(b) and 7(c) (neural network approximation error vs width or depth):

```
python train_width.py
python train_depth.py
```

Create a 3D plot of neural network fitting result and reproduce Figure 8 (solution slices):

```
python train_2d_plot.py
```
Figures are saved in the directory `figures/`. 