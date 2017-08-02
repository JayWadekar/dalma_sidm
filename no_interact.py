#!/usr/bin/env python

#SBATCH -p serial

#SBATCH -o "/scratch/dsw310/sidm_data/chi_saved/output.log"
#SBATCH -e "/scratch/dsw310/sidm_data/chi_saved/error.log"
#SBATCH -n 28
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=dsw310
#SBATCH --time=04:00:00
# Just calculating column depth 

import os
import numpy as np
import matplotlib.pyplot as plt
import galpy
import galpy.potential
from pylab import *
import random
import math
import time
import multiprocessing
import scipy
from scipy.interpolate import interp1d

sys.path.append(os.getcwd())

from galpy.potential import MiyamotoNagaiPotential, NFWPotential, PowerSphericalPotentialwCutoff, MWPotential2014, vcirc
from galpy.orbit import Orbit
from galpy.util import bovy_conversion

chi_total = []
o_out = []
chiorb = []
xy = []
t_tot = 10./bovy_conversion.time_in_Gyr(220.,8.)
n_t = 6000.

#mp= MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=.6)
#nfw= NFWPotential(a=16/8.,normalize=.35)
#bp= PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=0.05)
mwp = MWPotential2014
ts= np.linspace(0,t_tot,n_t) 
cst = 1.989*220.*3.154/pow(3.086,3)*bovy_conversion.time_in_Gyr(220.,8.)*t_tot/n_t
csthalo = 0.22*3.154*1.67*1.35*pow(5,-1.5)*bovy_conversion.time_in_Gyr(220.,8.)*t_tot/n_t
cst_d = 2*890./(4*4096*np.pi)*1.989*220.*3.154/pow(3.086,3)*bovy_conversion.time_in_Gyr(220.,8.)*t_tot/n_t/2.0217 #c=18.5

#Constructing the gas distributions
gas_dens_data = np.loadtxt("/scratch/dsw310/sidm_source/Input_data/hI_dens_galpy.dat") #M_sun/pc^3
gas_dens_log = interp1d(gas_dens_data[:,0], np.log(gas_dens_data[:,1]),fill_value='extrapolate')
gas_scale_data = np.loadtxt("/scratch/dsw310/sidm_source/Input_data/hI_height_galpy.dat")#kpc
gas_scale_log = interp1d(gas_scale_data[:,0], np.log(gas_scale_data[:,1]),fill_value='extrapolate')

#%%
''' Converting SMILE output to galpy input  (b4 for resolving binding energy error)
lines = np.load("/scratch/dsw310/sidm_source/Input_data/nfw_MW_23.npy")
galpy_convert= []
vc = np.sqrt(6.67*1.989/3.086)*1000 #N-body units G=M(ref_smile)=R (ref_smile) = 1

for l in lines:
    r = np.sqrt(l[0]**2 + l[1]**2)
    z = l[2]
    vz = l[5]*vc
    vr = (l[3]*l[0]+l[4]*l[1])*vc/r
    vt = (-l[3]*l[1]+l[4]*l[0])*vc/r
    phi = np.arctan2(l[1], l[0])
    galpy_convert.append([r/8.,vr/220.,vt/220.,z/8.,vz/220.,phi])
    
np.save("/scratch/dsw310/sidm_source/Input_data/converted_MW_23.npy",galpy_convert)
'''
#%%

''' For equilibrating the halo
lines = np.load("/scratch/dsw310/sidm_source/Input_data/equilibrated_MW_23.npy")
t_tot = 4./bovy_conversion.time_in_Gyr(220.,8.)
n_t = 2000.
ts= np.linspace(0,t_tot,n_t) 
def eqb(i):
    o= Orbit(vxvv=lines[i]) #Initial condition [R,vR,vT,z,vz,phi] #Cylindrical system
    o.integrate(ts,MWPotential2014,method='dopr54_c')
    return([o.R(ts[-1]),o.vR(ts[-1]),o.vT(ts[-1]),o.z(ts[-1]),o.vz(ts[-1]),o.phi(ts[-1])])
j=np.arange(0,len(lines),1)
if __name__ == '__main__':
    p = multiprocessing.Pool(28)
    eqborb = p.map(eqb, j)
np.save("/scratch/dsw310/sidm_source/Input_data/equilibrated_MW_23_2.npy",eqborb)
'''


#''' For column density of sampled orbits
lines = np.load('/scratch/dsw310/sidm_source/Input_data/equilibrated_MW_23.npy')

def coldep(i):
    column_b = 0. 
    column2_b = 0. #just gas halo
    #column_d = 0. #just DM halo
    radavg = 0.
    radmax = 0.
    sphradmax = 0.
    z2 = 0. 
    o= Orbit(vxvv=lines[i]) #Initial condition [R,vR,vT,z,vz,phi] #Cylindrical system
    o.integrate(ts,MWPotential2014,method='dopr54_c')
    for t in ts:
        column_b+= (np.exp(gas_dens_log(o.R(t)*8.)-pow(o.z(t)*8.,2)/np.exp(gas_scale_log(o.R(t)*8)))*np.linalg.norm(np.array([o.vR(t),o.vT(t)-vcirc(mwp,o.R(t)),o.vz(t)]))*cst) + (pow(1+(o.r(t)/(5./8.))**2,-0.75)*np.linalg.norm(np.array([o.vR(t),o.vT(t)-183/220.,o.vz(t)]))*csthalo)
        column2_b+=pow(1+(o.r(t)/(5./8.))**2,-0.75)*np.linalg.norm(np.array([o.vR(t),o.vT(t)-183/220.,o.vz(t)]))*csthalo
        #column_d+=cst_d/o.r(t)/pow(1+o.r(t)/2.,2)*np.linalg.norm(np.array([o.vR(t),o.vT(t),o.vz(t)])) 
        radavg+=o.R(t)
        z2+=pow(o.z(t),2)
        if radmax < o.R(t):
            radmax = o.R(t)
        if sphradmax < o.r(t):
            sphradmax = o.r(t)
    radavg *=8.
    z2 *=64.
    radmax *=8.
    sphradmax *=8.
    #return([(i+1),o.E(),radavg/len(ts),np.sqrt(z2/len(ts)),column,column2,o.zmax()*8.,radmax,sphradmax]) # 9 cols, don't delete
    return([(i+1),column_b,column2_b,radavg/len(ts),np.sqrt(z2/len(ts)),o.zmax()*8.,radmax,sphradmax,o.E()])


j=np.arange(0,900000,200)
if __name__ == '__main__':
    p = multiprocessing.Pool(28)
    chiorb = p.map(coldep, j)
    p.close()
    p.join()
                
np.savetxt("/scratch/dsw310/sidm_data/chi_saved/chi_23.dat",chiorb)
#'''

''' For individual orbit analysis
num = 10000
column=0.
lines = np.loadtxt("equilibrated_MW.dat", comments="#", unpack=False)
print lines[num]
o= Orbit(vxvv=lines[num]) #Initial condition [R,vR,vT,z,vz,phi] #Cylindrical system
o.integrate(ts,MWPotential2014,method='odeint')
for t in ts:
    column+= (mp.dens(o.R(t),o.z(t))+bp.dens(o.R(t),o.z(t)))*np.linalg.norm(np.array([o.vR(t),o.vT(t)-vcirc(mwp,o.R(t)),o.vz(t)]))*cst
print column

'''

'''
#Old Code
num = 7000
column=0.
temp = 0. # for xy plane crossing
lines = np.loadtxt("converted_MW.dat", comments="#", unpack=False)
print lines[num]
o= Orbit(vxvv=lines[num]) #Initial condition [R,vR,vT,z,vz,phi] #Cylindrical system
#o= Orbit(vxvv=[1.49308, 0.400345, -0.0111328, -0.122422, 0.481932, -1.36294])
o.integrate(ts,MWPotential2014,method='odeint')
for t in ts:
    column+= mp.dens(o.R(t),o.z(t))*bovy_conversion.dens_in_msolpc3(220.,8.)*np.linalg.norm(np.array([o.vR(t),o.vT(t)-mp.vcirc(o.R(t)),o.vz(t)]))*cst/5
    o_out.append([o.x(t)*8.,o.y(t)*8.,o.z(t)*8.,o.vx(t)*220.,o.vy(t)*220.,o.vz(t)*220.]) #kpc and km/s
    if ((o.z(t)*temp)<0):
        xy.append([o.x(t)*8.,o.y(t)*8.])
    temp = o.z(t)   
print column
  
np.savetxt("./Saved_data/noout_xy/"+str(num)+"noout.dat",o_out)
np.savetxt("./Saved_data/noout_xy/"+str(num)+"noxy.dat",xy)
'''
