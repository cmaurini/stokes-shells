from dolfin import *
comm = MPI.comm_world 
import numpy as np
from math import pi, sin, cos, sqrt
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nproc = MPI.size(comm)
mpiRank = MPI.rank(comm)
print('MPI PROCESS RANK ', mpiRank)
parameters["ghost_mode"] = "shared_facet"
parameters["refinement_algorithm"] = "plaza"
parameters["mesh_partitioner"] = "ParMETIS"
parameters["partitioning_approach"] = "PARTITION"


outdir = 'phidot/'
omega = Constant(1.)
alphadot = Constant(1.)

def mark(alpha, indicators):
    # Sort eta_T in decreasing order and keep track of the cell numbers
    #etas = indicators.vector().array()
    etas = indicators.vector().get_local()
    indices = etas.argsort()[::-1]
    sorted = etas[indices]
    
    # Compute sum and fraction of indicators
    total = sum(sorted)
    fraction = alpha*total

    t_total= MPI.sum(comm,total)
    t_fraction= alpha*t_total
    if (mpiRank == 0):
        print('\n\nTotal error is %f, fraction is %f with given alpha value %f\n'%(t_total,t_fraction,alpha))  
    
    # Define cell function to hold markers
    mesh = indicators.function_space().mesh()
    markers = MeshFunction("bool", mesh, 3, False)
    #if (mpiRank == 0):
    #    print('\n\nMarkers size is %i\n'%(markers.size()))
    # Iterate over the cells
    nmarked = 0
    v = 0.0
    
    for i in indices:
        # Stop if we have marked enough
        if v >= fraction or (total/t_total < 1./2./nproc):
            break
        # Otherwise
        nmarked = nmarked+1
        markers.array()[i] = True
        v += sorted[i]

    #print('Marked %i cells on processor %i'%(nmarked,MPI.rank(comm)))
    ntot = MPI.sum(comm,nmarked)
    if (mpiRank == 0):
        print('\n\nMarked %i cells\n'%(ntot))

    #print('Error on processor %i is %f, fraction is %f, marked %i cells\n'%(mpiRank,total,fraction,nmarked))  
    return markers



def solve_stokes(mesh, c, cname, ii):
    if (mpiRank == 0):
        print('c value is %f'%(c))  
    ncells = MPI.sum(comm,mesh.num_cells())
    if (mpiRank == 0):
        print('\n\nStarting Stokes flow solution with: %i cells.\n'%(ncells))
        print('Mesh hmax is: %f'%(mesh.hmax()))
        print('Mesh hmin is: %f'%(mesh.hmin()))
    # Boundaries
    class Surface(SubDomain):
        def inside(self, x, on_boundary):
            return (x[0]>-1.5 and x[0]<1.5) and (x[1]>-1.5 and x[1]<1.5) and (x[2]>-1.5 and x[2]<1.5) and on_boundary
    class Outer(SubDomain):
        def inside(self, x, on_boundary):
            return ((x[0]**2+x[1]**2+x[2]**2)**0.5 > 1.75)  and on_boundary
    surface = Surface()
    outer = Outer()
    domains = MeshFunction("size_t", mesh,2)
    domains.set_all(0)
    surface.mark(domains, 1)
    outer.mark(domains, 1000)

    alphadot = Constant(1.)  
    vel_p = Expression(("-c*x[2]*x[1]","x[0]-1/c*(asin( c*x[0]/sqrt( pow(c*x[0],2) + pow((1-c*x[2]),2) )) )", "c*x[0]*x[1]"),c=c, degree=2)

    #---------------------------SOLVE STOKES--------------------------------------------------------------------------------------
    element_u = VectorElement("CG", mesh.ufl_cell(), 2)
    element_p = FiniteElement("CG", mesh.ufl_cell(), 1)
    element = MixedElement([element_u,element_p])
    V = FunctionSpace(mesh,element)
    V1 = V.sub(0).collapse()
    fve = interpolate(vel_p,V1)     #imposed velocity field for later projection

    fields = Function(V)
    trialfields = TrialFunction(V)
    testfields = TestFunction(V)

    (u, p) = split(trialfields)
    (v, q) = split(testfields)

    noslip = Constant((0.0, 0.0, 0.0))
    bc1 = DirichletBC(V.sub(0), vel_p, domains, 1)
    bc2 = DirichletBC(V.sub(0), noslip, domains, 1000)
    # Collect boundary conditions
    bcs = [bc1, bc2]

    # Define variational problem
    K = Constant(100)
    f = Constant((0.0, 0.0, 0.0))
    nu = Constant(1.)
    a = nu*inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx 
    L = inner(f, v)*dx

    # Form for use in constructing preconditioner matrix
    b = nu*inner(grad(u), grad(v))*dx + p*q*dx

    #set_log_level(13)
    # Assemble system
    A, bb = assemble_system(a, L, bcs)

    # Assemble preconditioner system
    P, btmp = assemble_system(b, L, bcs)

    # Create Krylov solver and AMG preconditioner
    solver = KrylovSolver("minres", "amg")
    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

    # Solve
    if (mpiRank == 0):
        print("\nStarting Stokes flow solution")
    start = time.time()
    U = Function(V)
    solver.solve(U.vector(), bb)
    if (mpiRank == 0):
        print("Finished second solve in %f seconds"%(time.time()-start))

    # Get sub-functions
    u, p = U.split(deepcopy=True)

    u.rename('u','u')
    p.rename('p','p')

    if (mpiRank == 0):
        print("Writing solutions")

    # Save solution in XDMF format
    with XDMFFile(MPI.comm_world, outdir+'velocity_c_'+str(int(c*100))+'.xdmf') as file:
        file.write(u)
    with XDMFFile(MPI.comm_world, outdir+'pressure_c_'+str(int(c*100))+'.xdmf') as file:
        file.write(p)

    #-------------INTEGRATE FORCES/TORQUES-------------------------------------------------------------------
    if (mpiRank == 0):
        print("Integrating forces/torques")
    I = Identity(mesh.geometry().dim())
    n = FacetNormal(mesh)

    ds = Measure("ds", subdomain_data=domains)

    D = 0.5*(grad(u)+grad(u).T)  # or D=sym(grad(v))
    T = p*I + 2*nu*D
    force = T*n
    Dx = force[0]*ds(1)
    Dy = force[1]*ds(1)
    Dz = force[2]*ds(1)
    [Fx,Fy,Fz]=[assemble(Dx),assemble(Dy),assemble(Dz)]
    if (mpiRank == 0):
        print('Force:')
        print([Fx,Fy,Fz])

    #Torque
    x = SpatialCoordinate(mesh)
    Torque = cross(x,force)
    [Tx,Ty,Tz] = [assemble( Torque[i]*ds(1) ) for i in range(3)]
    if (mpiRank == 0):
        print('Torque')
        print([Tx,Ty,Tz])

    projection_phidot = inner(force,vel_p)*ds(1)
    prj = assemble(projection_phidot)
    if (mpiRank == 0):
        print("\nValue of projection on precession velocity field is %f\n"%(prj))

    if (mpiRank == 0):
        print([c, prj, Fx, Fy, Fz, Tx, Ty, Tz])

    res = [c, prj, Fx, Fy, Fz, Tx, Ty, Tz]
    return u,p,res

alpha_ref=0.0035
if (mpiRank == 0):
    print('\nSOLVING CASE FOR PRECESSION, FOR ALPHA=%f\n'%(alpha_ref))  

cs = [0.05, 0.1,  0.15,  0.2,  0.25,  0.3,  0.4,  0.5, 0.6, 0.7,  0.8,  0.9,  1.,  1.1, 1.2, 1.3, 1.4, 1.5, 1.5708, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.3562, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1]
cs_name = ['5','10','15','20','25','30','40','50','60','70', '80', '90', '100', '110', '120', '130', '140', '150', '157', '170', '180', '190', '200',  '210', '220', '229', '235', '250', '260', '270','280','290','300','310']

restot = []
for c,cn in zip(cs,cs_name):
    mesh=Mesh(MPI.comm_world)
    filename = 'bfile_alpha_c_'+cn+'.h5'
    if (mpiRank == 0):
        print('Reading file ' + filename)

    hdf = HDF5File(mesh.mpi_comm(), filename, "r")
    hdf.read(mesh, "/mesh", False)
    subdomains = MeshFunction("size_t", mesh, 3)
    hdf.read(subdomains, "/subdomains")

    domains = MeshFunction("size_t", mesh, 2)
    hdf.read(domains, "/boundaries")

    ncells = MPI.sum(comm,mesh.num_cells())
    if (mpiRank == 0):
        print('\n\nRead new mesh, number of cells: %i.\n'%(ncells))

    res = []
    dres= []
    u,p,resi=solve_stokes(mesh,c,cn,0)
    ncells = MPI.sum(comm,mesh.num_cells())
    resi = [ncells] + resi
    res.append(resi)
    if (mpiRank == 0):
        print('\nSolved over starting mesh. Start refinement iteration\n')
    value_old=resi[8]
    ii=1

    while True:
        if (mpiRank == 0):
            print('\n\n-------------------------------------------------Iteration %i'%(ii))

        mesh.init(1, 2) # Initialise facet to cell connectivity
        # Get the cell sizes:
        h = CellDiameter(mesh)

        # Define the element residual:
        r = div(grad(u)) + grad(p) 

        # Define the jump residual
        n = FacetNormal(mesh)
        I = Identity(mesh.geometry().dim())
        j = jump(grad(u)+p*I, n)

        # Create a DG0 function space (piecewise constants, no continuity)
        DG0 = FunctionSpace(mesh, "DG", 0)

        # Create a test function on this space
        w = TestFunction(DG0)

        # Define the error indicator form
        E_T = h**2*dot(r,r)*w*dx + 2*avg(w)*avg(h)*dot(j,j)*dS + div(u)**2*w*dx

        # Define a function to hold the result of the evaluation
        indicators = Function(DG0)

        # Assemble the error indicator form into the coefficient vector
        assemble(E_T, tensor=indicators.vector())
        #File("indicator"+str(ii)+".pvd") << indicators

        markers = mark(alpha_ref, indicators)
        mesh = refine(mesh, markers, True)
        ncells = MPI.sum(comm,mesh.num_cells())
        if (mpiRank == 0):
            print('\nNew mesh has %i elements.'%(ncells))
        u,p,resi = solve_stokes(mesh,c,cn,ii)

        resi = [ncells] + resi
        res.append(resi)
        value = resi[8]
        diff = np.abs((value-value_old)/value_old)
        dres.append(diff)

        value_old=value


        if (diff < 0.01):
            if (mpiRank == 0):
                print('\n\nTarget value difference lower than 0.01, interrupting at iteration %i, number of cells was %i\n'%(ii,ncells))
            break
        if (ncells > 3.0e6):
            if (mpiRank == 0):
                print('\n\nMax number of cells (3.0*1e6) exceeded, interrupting at iteration %i, difference in target value was %f \n'%(ii,diff))
            break
        if (mpiRank == 0):
            print('\nVariation in target value is %f (>0.01), number of cells is %i. Continuing with next iteration.\n'%(diff,ncells))
        ii = ii+1
    if (mpiRank == 0):
        print(res)
    if (mpiRank == 0):
        print('\n',dres)
    restot.append(res[-1])

if (mpiRank == 0):
    print(restot)
quit()
