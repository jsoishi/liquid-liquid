"""
Sciortino et al PRE 1993 

Usage:
    sol-gel.py <config_file>

"""
import os
import time
import logging
from configparser import ConfigParser
import numpy as np
import pathlib

import dedalus.public as de

from dedalus.tools  import post
from dedalus.extras import flow_tools

from filter_field import filter_field

logger = logging.getLogger(__name__)

from docopt import docopt

# parse arguments
args = docopt(__doc__)
filename = pathlib.Path(args['<config_file>'])
# Parse .cfg file to set global parameters for script
config = ConfigParser()
config.read(str(filename))
logger.info("Using config file {}".format(filename))

# parameters
params = config['params']

L  = params.getfloat('L')
nx = params.getint('nx')
ny = params.getint('ny')

ampl = params.getfloat('ampl')

Mc    = params.getfloat('Mc')
T     = params.getfloat('T')
c0    = params.getfloat('c0')
phis = params.getfloat('phis')
ΔF    = params.getfloat('ΔF')
run_opts = config['run']
dt = run_opts.getfloat('dt')

x = de.Fourier('x',nx,interval=[0, L], dealias=2.)
y = de.Fourier('y',nx,interval=[0, L], dealias=2.)

domain = de.Domain([x, y], grid_dtype='float')

basedir = pathlib.Path('scratch')
outdir = "CH_" + filename.stem
data_dir = basedir/outdir
if domain.dist.comm.rank == 0:
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

ch = de.IVP(domain, variables=['ψ','c'])

ch.parameters['Mc']    = Mc   
ch.parameters['ε']     = (1 - T)
ch.parameters['c0']    = c0   
ch.parameters['phis']  = phis
ch.parameters['ΔF']    = ΔF # in units of kT
logger.info("Running with Mc = {}, ε = {}, c0 = {}, phis = {}, ΔF = {}".format(Mc,1-T, c0, phis, ΔF))
ch.parameters['p'] = np.exp(ΔF)/(1 + np.exp(ΔF))

ch.substitutions["Mp"] = "exp(-c/c0)"
ch.substitutions["phi"] = "0.5*(ψ + 1)"
ch.substitutions["g"] = "(p*phi - phis)/(1 - phis)"
ch.substitutions["Lap(A)"] = "dx(dx(A)) + dy(dy(A))"
ch.substitutions["DLap(A)"] = "dx(dx(dx(dx(A)))) + 2*dx(dx(dy(dy(A)))) + dy(dy(dy(dy(A))))"
ch.substitutions["dot(Ax, Ay, Bx, By)"] = "Ax*Bx + Ay*By"

ch.add_equation("dt(ψ) =  -Mp*DLap(ψ) + Mp*Lap(ψ**3 - ε*ψ) - dot(dx(Mp),dy(Mp), dx(Lap(ψ)), dy(Lap(ψ))) - dot(dx(Mp), dy(Mp), dx(ε*ψ - ψ**3), dy(ε*ψ - ψ**3))")
ch.add_equation("dt(c) = Mc*(g*c - c**2)")

solver = ch.build_solver(de.timesteppers.RK222)

analysis_tasks = []
snap = solver.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), iter=100, max_writes=200)
snap.add_task("ψ")
snap.add_task("ψ", name='ψ_c', layout='c')
snap.add_task("c")
snap.add_task("c", name='c_c', layout='c')
analysis_tasks.append(snap)

#ts = solver.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'), iter=10)

# Create initial conditions
gshape = ch.domain.dist.grid_layout.global_shape(scales=ch.domain.dealias)
slices = ch.domain.dist.grid_layout.slices(scales=ch.domain.dealias)
rand = np.random.RandomState(seed=100)
noise = rand.standard_normal(gshape)[slices]
nzeros = int((nx*ny)/2)
#noise = np.array([-1]*nzeros + [1]*nzeros)
#np.random.shuffle(noise)
#noise = noise.reshape(nx,ny)[slices]

def circle(x, x0, r, eps):
    rr = np.sqrt((x[0] - x0[0])**2 + (x[1] - x0[1])**2)
    
    output = np.tanh((rr - r)/(np.sqrt(2)* eps))
    return output
    
ψ0 = solver.state['ψ']
c0 = solver.state['c']
ψ0.set_scales(ch.domain.dealias, keep_data=False)
c0.set_scales(ch.domain.dealias, keep_data=False)
#x0 = [0.5, 0.5]
#a = 0.25
#width = np.sqrt(ε/γ)
#ψ0['g'] = circle(domain.grids(), x0, a, width)


# 1-D tanh profile
# x = domain.grid(0)
# x0 = L/4.
# x1 = 3*L/4.
# width = 7.071*np.sqrt(κ/ρ)
# ψ0['g'] = np.tanh((x-x0)/width) + np.tanh((x1-x)/width) - 1

ψ0['g'] = ampl*noise
c0['g'] = 1e-12*rand.standard_normal(gshape)[slices]
filter_field(ψ0)#, frac=0.5)
filter_field(c0)#, frac=0.5)
ψ_min = ψ0['g'].min()
ψ_max = ψ0['g'].max()

# ψ0['g'] = 2*((ψ0['g'] - ψ_min)/(ψ_max - ψ_min)) - 1
# ψ0['g'] -= ψ0['g'].mean()

solver.stop_wall_time = run_opts.getfloat('stop_wall_time')
solver.stop_sim_time = run_opts.getfloat('stop_sim_time')

if run_opts.getint('stop_iteration'):
    solver.stop_iteration = run_opts.getint('stop_iteration')
else:
    solver.stop_iteration = np.inf

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("integ(ψ)", name="Cint")
flow.add_property("integ(c)", name="gint")
flow.add_property("Mp", name="Mp")
flow.add_property("integ(Mp)", name="Mp_int")
#flow.add_property("integ(f)", name="f")
#flow.add_property("integ(grad_en)", name="grad_en")

C0 = ψ0.integrate()['g'][0,0]
logger.info("Inital Concentration =  {:10.7e}".format(C0))

safety = 0.01
dx = L/nx
#dt = safety * dx**2/(M*ρ)
logger.info("Calculated dt = {:f}".format(dt))
if run_opts.getfloat('dt'):
    dt = run_opts.getfloat('dt')
logger.info("Running with dt = {:f}".format(dt))

start  = time.time()
while solver.ok:
    solver.step(dt)
    if (solver.iteration-1) % 10 == 0:
        logger.info("Step {:d}, Time {:f}".format(solver.iteration,solver.sim_time))
        logger.info("Integrated Concentration = {:10.7e}".format(flow.max("Cint")))
        logger.info("Integrated gelation = {:10.7e}".format(flow.max("gint")))
        logger.info("Integrated Mp = {:10.7e}".format(flow.max("Mp_int")))
        logger.info("Maximum Mp = {:10.7e}".format(flow.max("Mp")))
        logger.info("Minimum Mp = {:10.7e}".format(flow.min("Mp")))
        #logger.info("f = {:10.7e}; 0.5 κ |∇phi|² = {:10.7e}".format(flow.max("f"), flow.max("grad_en")))

stop = time.time()

logger.info("Total Run time: {:5.2f} sec".format(stop-start))
logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
