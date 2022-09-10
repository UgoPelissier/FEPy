# FEPy - Finite Element (Python)

Python project to build the Delaunay triangulation of 2D points and solve the Poisson and convection-diffusion equations in different study cases (Cavity and Oven).

Oven study case is inspired from a study case proposed during a class, which was initially solved using the open-source PDE solver *FreeFem++*.

The utlimate goal was to implement from scratch both a mesher and a PDE solver that could be used for this case. Along the way, additional features were developped such as time-dependant calculations and adaptive meshing.

## Study case: Oven heating

![image](https://user-images.githubusercontent.com/95024044/189478083-f625327c-88c6-4133-bb82-6e5045aa715c.png)

We will note $\Omega$ the opening representing the oven. The oven is a square of 1 m side, and the room is a rectangle of 1 m by 0.4 m, in the center of the oven (to simplify, we will take the origin in the center of the square). We will note $C_i$ the resistors (modeled here as circles of radius 0.05), and placed at the points (±0.75, ±0.75) (initially, $N_r$ = 4).
The upper edge of the furnace $\Gamma_u$ is assumed to be maintained at $T_u = 50 \degree C$, the lower edge $\Gamma_d$ at $T_d = 10 \degree C$, and the two lateral edges $\Gamma_l$ are isolated (zero heat flux). The temperature is thus a solution of the equation

$$
\begin{align*}
-\text{div}(k \ \text{grad}T) & = f & \text{dans} \ \Omega \\
T & = T_u & \text{sur} \ \Gamma_u \\
T & = T_d & \text{sur} \ \Gamma_d \\
k\frac{\partial T}{\partial n} & = 0 & \text{sur} \ \Gamma_l
\end{align*}
$$

In this equation, $k$ is the thermal diffusion coefficient, which is variable in the oven: it is 1 in the room, and 10 in the rest of the oven. $f$ is the heat source, and will be of the form $f = \sum\limits_{i=1}^{N_r}\alpha_i{1}_{c_i}$, where the $\alpha_i$ coefficients are either chosen (in the straightforward problem), or to be determined (in the inverse problem).

### Mesh

Delaunay triangulation
- Unstructured
- Constrained
- Maximum mesh size
- Minimum angle of 20°

![image](https://user-images.githubusercontent.com/95024044/189440875-f7736f4b-e30b-4d10-aac1-c24f3ba526b9.png)

![image](https://user-images.githubusercontent.com/95024044/189417475-4773a97a-d9ab-4cac-9157-da677e53ad33.png)

### PDE Solver

The variational formulation of the problem is as follows

$$
\begin{align*}
 T & = T_u & \text{sur} \ \Gamma_u \\
 T & = T_d & \text{sur} \ \Gamma_d \\
 \int_{\Omega} k \nabla(T).\nabla(v) & = \int_{\Omega} fv & \text{for all} \ v \in H^{1}(\Omega) \ \text{such that:} \ v=0 \ \text{sur} \ (\Gamma_u\cup\Gamma_d)
\end{align*}
$$

![image](https://user-images.githubusercontent.com/95024044/189478517-c40aa26d-c1b2-4915-9b18-e39e22b5b8e1.png)

![image](https://user-images.githubusercontent.com/95024044/189478753-23563b2e-f0c2-433d-ac6d-9e0d1f653703.png)
