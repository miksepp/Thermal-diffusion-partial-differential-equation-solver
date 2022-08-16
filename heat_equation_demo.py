import fenics as fn

# Create mesh and define function space
mesh = fn.Mesh("demomesh.xml")
subdomains = fn.MeshFunction("size_t", mesh, "demomesh_physical_region.xml")
boundaries = fn.MeshFunction("size_t", mesh, "demomesh_facet_region.xml")
V = fn.FunctionSpace(mesh, 'Lagrange', 1)
ds = fn.ds(subdomain_data=boundaries)

# Define boundary conditions for the top and bottom edges
T_bot = 280
bc_bot = fn.DirichletBC(V, T_bot, boundaries, 2)
T_top = 320
bc_top = fn.DirichletBC(V, T_top, boundaries, 1)
bcs = [bc_bot, bc_top]

# Define initial condition
initial_condition = 300
T_i = fn.project(initial_condition, V)

# Define variables
T = fn.TrialFunction(V)
v = fn.TestFunction(V)
a = 1

# Define time steps
t_end = 10
t = 0
num_steps = 1000
dt = t_end/num_steps

# Define variational form
integrand1 = (T *v + dt * a * fn.dot(fn.grad(T), fn.grad(v))) * fn.dx
integrand2 = T_i * v * fn.dx
T = fn.Function(V)

# Solve the equation and save to a vtk-file for each time step
vtkfile = fn.File('heat_equation_demo/solution.pvd')
for n in range(num_steps):
    fn.solve(integrand1 == integrand2, T, bcs)
    T_i.assign(T)
    vtkfile << (T, t)
    t += dt
