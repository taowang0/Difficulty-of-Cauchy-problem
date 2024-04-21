## Understanding the Difficulty of Solving Cauchy Problems with PINNs 

This repo contains the code for experiments in paper ["Understanding the Difficulty of Solving Cauchy Problems with PINNs"](https://[github.com/sachabinder/Burgers_equation_simulation/tree/main](https://github.com/taowang0/Understanding-the-Difficulty-of-Solving-Cauchy-Problems-with-PINNs/edit/main/README.md)) published at L4DC, 2024.

#### True solution by spectral methods

The true solution data (involved in generating Figure 6(d) and 7(a)) is obtained by the solver from 
[https://github.com/sachabinder/Burgers\_equation\_simulation/tree/main](https://github.com/sachabinder/Burgers_equation_simulation/tree/main).

Run solver:

```
python Burgers_solver_SP.py
```


#### Solving with PINN loss

Reproduce Figure 6(a)-(c) (failures by minimizing PINN loss):

```
python Solving_with_PINN_loss.py
```

#### Neural network fitting (for MSE loss)

Reproduce Figure 7(b) and 7(c) (neural network approximation error vs width or depth):

```
python train_width.py
python train_depth.py
```

Create a 3D plot of neural network fitting result and reproduce Figure 8 (solution slices):

```
python train_2d_plot.py
```
