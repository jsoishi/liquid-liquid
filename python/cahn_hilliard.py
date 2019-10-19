"""Test script to do Cahn_hilliard

"""
import os
import time
import logging
import numpy as np

import dedalus.public as de

from dedalus.tools  import post
from dedalus.extras import flow_tools

from filter_field import filter_field

logger = logging.getLogger(__name__)

# parameters
L = 200#1
nx = 200#128
ny = 200#128

ampl = 1e-4

# # diffusion coeff D = (λ γ1)/ε²
# D = L**2
# dx = L/nx
#γ = 1
#ε = 5e-4

κ = 2.
ρₛ = 5.
M = 5.

ρ = 2*ρₛ
data_dir = 'data'

x = de.Fourier('x',nx,interval=[0, L], dealias=2.)
y = de.Fourier('y',nx,interval=[0, L], dealias=2.)

domain = de.Domain([x, y], grid_dtype='float')

ch = de.IVP(domain, variables=['phi'])
# ch.parameters['γ'] = γ
# ch.parameters['ε']  = ε
ch.parameters['M'] = M
ch.parameters['κ'] = κ
ch.parameters['ρ'] = ρ

ch.substitutions["f"] = "ρ*(phi + 1)**2 * (1 - phi)**2"
ch.substitutions["grad_en"] = "0.5*κ*(dx(phi)**2 + dy(phi)**2)"
ch.substitutions["Lap(A)"] = "dx(dx(A)) + dy(dy(A))"
ch.substitutions["DLap(A)"] = "dx(dx(dx(dx(A)))) + 2*dx(dx(dy(dy(A)))) + dy(dy(dy(dy(A))))"
#ch.add_equation("dt(phi) + M*κ*DLap(phi) + M*ρ*Lap(phi) = M*ρ*Lap(phi**3)")
ch.add_equation("dt(phi) + M*κ*DLap(phi) = M*ρ*Lap(phi**3 - phi)")

solver = ch.build_solver(de.timesteppers.SBDF2)

analysis_tasks = []
snap = solver.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), iter=100, max_writes=200)
snap.add_task("phi")
snap.add_task("phi", name='phi_c', layout='c')
analysis_tasks.append(snap)

ts = solver.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'), iter=10)
ts.add_task("integ(f + grad_en)", name='free energy')
ts.add_task("integ(f)", name='chemical free energy')
ts.add_task("integ(grad_en)", name='gradient free energy')

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
    
phi0 = solver.state['phi']
#x0 = [0.5, 0.5]
#a = 0.25
#width = np.sqrt(ε/γ)
#phi0['g'] = circle(domain.grids(), x0, a, width)

# NIST benchmark
c0 = 0
ε = 0.01
x, y = domain.grids()
#phi0['g'] = c0 + ε*(np.cos(0.105*x)*np.cos(0.11*y)+(np.cos(0.13*x)*np.cos(0.087*y))**2+np.cos(0.025*x-0.15*y)*np.cos(0.07*x-0.02*y))
def fix_coeff(k, L):
    n = int(k*L/(2*np.pi))
    k_new = 2*np.pi * n/L
    return k_new
a0 = fix_coeff(0.105,L)
b0 = fix_coeff(0.11,L)
c0 = fix_coeff(0.13,L)
d0 = fix_coeff(0.087,L)
e0 = fix_coeff(0.025,L)
f0 = fix_coeff(0.15,L)
g0 = fix_coeff(0.07,L)
h0 = fix_coeff(0.02,L)
phi0['g'] = c0 + ε*(np.cos(a0*x)*np.cos(b0*y)+(np.cos(c0*x)*np.cos(d0*y))**2+np.cos(e0*x-f0*y)*np.cos(g0*x-h0*y))
#filter_field(phi0)

# 1-D tanh profile
# x = domain.grid(0)
# x0 = L/4.
# x1 = 3*L/4.
# width = 7.071*np.sqrt(κ/ρ)
# phi0['g'] = np.tanh((x-x0)/width) + np.tanh((x1-x)/width) - 1

# phi0 = solver.state['phi']
# phi0['g'] = ampl*noise
# filter_field(phi0)#, frac=0.5)
# phi_min = phi0['g'].min()
# phi_max = phi0['g'].max()

# phi0['g'] = 2*((phi0['g'] - phi_min)/(phi_max - phi_min)) - 1
# phi0['g'] -= phi0['g'].mean()

solver.stop_wall_time = 24*3600
solver.stop_sim_time = 8000.
solver.stop_iteration = np.inf#100

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("integ(phi)", name="Cint")
flow.add_property("integ(f)", name="f")
flow.add_property("integ(grad_en)", name="grad_en")

C0 = phi0.integrate()['g'][0,0]
logger.info("Inital Concentration =  {:10.7e}".format(C0))

safety = 0.01
dx = L/nx
dt = safety * dx**2/(M*ρ)
logger.info("dt = {:f}".format(dt))
#dt = 0.04#1e-4

start  = time.time()
while solver.ok:
    solver.step(dt)
    if (solver.iteration-1) % 10 == 0:
        logger.info("Step {:d}, Time {:f}".format(solver.iteration,solver.sim_time))
        logger.info("Integrated Concentration = {:10.7e}".format(flow.max("Cint")))
        logger.info("f = {:10.7e}; 0.5 κ |∇ϕ|² = {:10.7e}".format(flow.max("f"), flow.max("grad_en")))

stop = time.time()

logger.info("Total Run time: {:5.2f} sec".format(stop-start))
logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
