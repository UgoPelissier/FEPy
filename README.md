# FEPy - Finite Element (Python)

Python project to build the Delaunay triangulation of 2D points and solve the Poisson and convection-diffusion equations in different study cases (Cavity and Oven).

Oven study case is inspired from a study case proposed during a class, which was initially solved using the open-source PDE solver *FreeFem++*.

The utlimate goal was to implement from scratch both a mesher and a PDE solver that could be used for this case. Along the way, additional features were developped such as time-dependant calculations and adaptive meshing.

## Mesh

Delaunay triangulation
- Unstructured
- Constrained
- Maximum mesh size
- Minimum angle of 20°

![image](https://user-images.githubusercontent.com/95024044/189440875-f7736f4b-e30b-4d10-aac1-c24f3ba526b9.png)

![image](https://user-images.githubusercontent.com/95024044/189417475-4773a97a-d9ab-4cac-9157-da677e53ad33.png)

## Poisson

