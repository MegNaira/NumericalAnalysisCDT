# Pollutant spread in Southern England from a fire in Southampton
Assignment of the second part of the course on Numerical Analysis by the MFC CDT.

This code uses the time independent Galerkin finite element method to find solutions of the diffusion-advection equation.

To find the numerical solution to this problem, please run the code staticdiffadv_las.py. Make sure that, in the folder where you're doing this, you also have the file librarystatic2d_wADV.py (present in this repository), as it contains functions used by staticdiffadv_las.py. Note that this also needs the folder las_grids, downloadable from here, which contains the finite element grids.

## Author
Arianna Ferrotti [email here](mailto:a.ferrotti@soton.ac.uk)

## Dependencies
* numpy
* matplotlib
* scipy

## Acknowledgments
Thank you to Hilary Weller and Ian Hawke, lecturers of the course.
