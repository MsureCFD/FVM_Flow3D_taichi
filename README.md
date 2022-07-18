# FVM_Flow3D_taichi
***FVM_Flow3D_taichi*** is a transient solver for ***incompressible flow*** of Newtonian fluids, written in the [Taichi programming language](https://github.com/taichi-dev/taichi). It is based on the ***finite volume method*** on a ***staggered Cartesian grid*** and uses the conservative form of the second-order ***Adams–Bashforth scheme***, thus it has ***second order accuracy*** in both time and space. The code involves a ***two-step predictor–corrector*** procedure to solve the Navier-stokes equation.
## Installation
[Taichi programming language](https://github.com/taichi-dev/taichi) is required. 

You can easily install Taichi with Python's package installer `pip`:

`pip install taichi`

then run the code using:

`ti FVM_Flow3D_taichi.py` or `python FVM_Flow3D_taichi.py`
## Usage
### Running control
`ntime = 50000` is the total time steps that you would like to simulate for.
 
`snap_n = 100` is the interval for data output, if it is 100, then the 3D flow data will be output for each 100 steps.

`dt = 0.001` is the time interval for the simualtion. ***Please pay attention to the CFL condition***.

### Mesh parameters
`n = ti.Vector([32,32,16])` is the number of cells in three directions, not including the ghost cells 0 and n+1.

`start = ti.Vector([0.0,0.0,0.0])` is the start point of the computatioanl domain in three directions.

`finish = ti.Vector([1.0,1.0,1.0])`  is the end point of the computatioanl domain in three directions.

### Fluid properties
`nu = 0.01` is the viscosity of the fluid.

`ro = 1000` is the density of the fluid.

`fbody_x = 0.0, fbody_y = 0.0, fbody_z = 0.0` are the body forces in three directions.

### Boundary condition (not complete and the periodic boundary needs to be corrected)
xm - western   boundary  
xp - eastern   boundary     
zm - southern  boundary       
zp - northern  boundary     
ym - bottom    boundary  
yp - top       boundary  

1 - no   slip  boundary  
2 - free slip  boundary  
0 - periodic   boundary  

### Output
The data will be output into the monitor folder. It is in the [Tecplot](https://www.tecplot.com/) format and can be read by [Tecplot](https://www.tecplot.com/) or [Paraview](https://www.paraview.org/) for CFD visulazition and analysis.

## Case1 - Cavity flow
Cavity flow is benchmark case used to validate the code first. To set the velocity of top lid, changing the top boundary condition:
```   for i, k in ti.ndrange((1, nx), (1,nz)):  
        if yp == 1:  
            vel[i,ny,k][0] = 2*1 - vel[i,ny-1,k][0]  
```
Then the top boundary has a velocity of 2m/s. The animation is:
![cavityFlow.gif](/CasesVisualization/cavityFlow.gif) 

The comparision with literature [Ghia et al.](https://www.sciencedirect.com/science/article/pii/0021999182900584) is:
