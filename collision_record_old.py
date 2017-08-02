#!/usr/bin/env python

#SBATCH -p serial

#SBATCH -o "/scratch/dsw310/sidm_data/surface/node1/output.log"
#SBATCH -e "/scratch/dsw310/sidm_data/surface/node1/error.log"
#SBATCH -n 28
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=dsw310
#SBATCH --time=00:08:00

import os
import numpy as np
import matplotlib.pyplot as plt
import galpy
import galpy.potential
from pylab import *
import random
import math
import multiprocessing
import time
import scipy
from scipy.interpolate import interp1d

sys.path.append(os.getcwd())
#module('load','anaconda/2-4.1.1')
os.system('module load anaconda/2-4.1.1')
os.system('source activate sidm-py')

from galpy.potential import MiyamotoNagaiPotential, NFWPotential, PowerSphericalPotentialwCutoff, MWPotential2014, vcirc, vesc
from galpy.orbit import Orbit
from galpy.util import bovy_conversion


column_ref = 1670/500*1.24/6.04
t_tot = 1./bovy_conversion.time_in_Gyr(220.,8.)
n_t = 900.
ts= np.linspace(0,t_tot,n_t) 
ts_excl= np.linspace(0,t_tot*10,n_t*8) #arbitrarily chosen 
sigmav = 8/np.sqrt(3)/220.

#mp= MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=.6)
# np= NFWPotential(a=16/8.,normalize=.35)
bp= PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=0.05)
mwp = MWPotential2014
#tfrac = np.arange(0.05,10.,0.05)
#tfrac = np.append(tfrac,[10.,11.])#last time not counted but choose high enough value
lines = np.load('equilibrated_MW.npy')
cst = 1.989*220.*3.154/pow(3.086,3)*bovy_conversion.time_in_Gyr(220.,8.)*t_tot/n_t
csthalo = 0.22*3.154*1.67*1.35*pow(5,-1.5)*bovy_conversion.time_in_Gyr(220.,8.)*t_tot/n_t
cst_excl = cst*10./8.
csthalo_excl = csthalo*10./8.
bovy_dens=bovy_conversion.dens_in_msolpc3(220.,8.)

#Constructing the gas distributions
gas_dens_data = np.loadtxt('hI_dens_galpy.dat') #M_sun/pc^3
gas_dens_log = interp1d(gas_dens_data[:,0], np.log(gas_dens_data[:,1]),fill_value='extrapolate')
gas_scale_data = np.loadtxt('hI_height_galpy.dat')#kpc
gas_scale_log = interp1d(gas_scale_data[:,0], np.log(gas_scale_data[:,1]),fill_value='extrapolate')


def particle(num):
    o= Orbit(vxvv=lines[num]) #Initial condition [R,vR,vT,z,vz,phi] 
    print "Num,R,z:", ([num,lines[num][0]*8., lines[num][3]*8.])
    column=0.
    column0 = -column_ref*np.log(random.random())
    time = 0.
    tpick = 0.
    numcoll = 0
    i_tf = 0
    surf= []
    db_tpick = 0.
    
    #See if the orbit should be excluded--------------------------
    o.integrate(ts_excl,MWPotential2014,method='dopr54_c')
    for t in ts_excl:
            column+= (np.exp(gas_dens_log(o.R(t)*8.)-pow(o.z(t)*8.,2)/np.exp(gas_scale_log(o.R(t)*8)))*np.linalg.norm(np.array([o.vR(t),o.vT(t)-vcirc(mwp,o.R(t)),o.vz(t)]))*cst_excl) + (pow(1+(o.r(t)/(5./8.))**2,-0.75)*np.linalg.norm(np.array([o.vR(t),o.vT(t)-183/220.,o.vz(t)]))*csthalo_excl)
            if (column > column0):
                numcoll+=1
                column = 0. 
                column0 = -column_ref*np.log(random.random())
                theta = np.arccos(2.*random.random() - 1.)
                phi = 2*np.pi*random.random()
                veldm = np.array([o.vR(t),o.vT(t),o.vz(t)])
                if (abs(o.z(t))>(2/8.) and o.r(t)>(3/8.)):
                    velb = np.array([np.random.normal(0,sigmav)*math.exp(-o.r(t)), 183/220.+np.random.normal(0,sigmav)*math.exp(-o.r(t)), np.random.normal(0,sigmav)*math.exp(-o.r(t))])
                else:
                    velb = np.array([np.random.normal(0,sigmav)*math.exp(-o.R(t)), vcirc(mwp,o.R(t))+np.random.normal(0,sigmav)*math.exp(-o.R(t)), np.random.normal(0,sigmav)*math.exp(-o.R(t))])
                vel = 1./3.*velb+2./3.*veldm-1./3.*np.linalg.norm(veldm-velb)*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
                velb_final = 1./3.*velb+2./3.*veldm+2./3.*np.linalg.norm(veldm-velb)*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])       
                time+=t*bovy_conversion.time_in_Gyr(220.,8.)
                surf.append([o.R(t)*8.,o.phi(t),o.z(t)*8.,velb_final[0]*220.,velb_final[1]*220.,velb_final[2]*220.,num,vesc(mwp,o.R(t))*220.,time])
                o= Orbit(vxvv=[o.R(t),vel[0],vel[1],o.z(t),vel[2],o.phi(t)])    
                break
                
            elif (t==ts_excl[-1]): #counts for no collision in the given time
                time+=ts_excl[-1]*bovy_conversion.time_in_Gyr(220.,8.)  
                o= Orbit(vxvv=[o.R(t),o.vR(t),o.vT(t),o.z(t),o.vz(t),o.phi(t)])
    
    #Normal orbit integration-----------------------------------------
    while (time<10.):
        o.integrate(ts,MWPotential2014,method='dopr54_c')
        if (numcoll>30):
            break
        for t in ts:
            column+= (np.exp(gas_dens_log(o.R(t)*8.)-pow(o.z(t)*8.,2)/np.exp(gas_scale_log(o.z(t)*8)))*np.linalg.norm(np.array([o.vR(t),o.vT(t)-vcirc(mwp,o.R(t)),o.vz(t)]))*cst) + (pow(1+(o.r(t)/(5./8.))**2,-0.75)*np.linalg.norm(np.array([o.vR(t),o.vT(t)-183/220.,o.vz(t)]))*csthalo)  
            if (column > column0):
                numcoll+=1
                column = 0. 
                column0 = -column_ref*np.log(random.random())
                theta = np.arccos(2.*random.random() - 1.)
                phi = 2*np.pi*random.random()
                veldm = np.array([o.vR(t),o.vT(t),o.vz(t)])
                if (abs(o.z(t))>(2/8.) and o.r(t)>(3/8.)):
                    velb = np.array([np.random.normal(0,sigmav)*math.exp(-o.r(t)), 183/220.+np.random.normal(0,sigmav)*math.exp(-o.r(t)), np.random.normal(0,sigmav)*math.exp(-o.r(t))])
                else:
                    velb = np.array([np.random.normal(0,sigmav)*math.exp(-o.R(t)), vcirc(mwp,o.R(t))+np.random.normal(0,sigmav)*math.exp(-o.R(t)), np.random.normal(0,sigmav)*math.exp(-o.R(t))])
                vel = 1./3.*velb+2./3.*veldm-1./3.*np.linalg.norm(veldm-velb)*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
                velb_final = 1./3.*velb+2./3.*veldm+2./3.*np.linalg.norm(veldm-velb)*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])  
                time+=t*bovy_conversion.time_in_Gyr(220.,8.)
                surf.append([o.R(t)*8.,o.phi(t),o.z(t)*8.,velb_final[0]*220.,velb_final[1]*220.,velb_final[2]*220.,num,vesc(mwp,o.R(t))*220.,time])
                o= Orbit(vxvv=[o.R(t),vel[0],vel[1],o.z(t),vel[2],o.phi(t)])    
                break
                
            elif (t==ts[-1]): #counts for no collision in the given time
                time+=ts[-1]*bovy_conversion.time_in_Gyr(220.,8.)   
                o= Orbit(vxvv=[o.R(t),o.vR(t),o.vT(t),o.z(t),o.vz(t),o.phi(t)])                                    
    return(surf)



#deleting prev output files
if (os.path.exists("/scratch/dsw310/sidm_data/surface/node1/coll.npy")==False):
    np.save("/scratch/dsw310/sidm_data/surface/node1/coll.npy",[])

if __name__ == '__main__':
    for k in range(4,5):
        j=np.arange((2*k),(2*(k+1)),1) 
        p = multiprocessing.Pool(28)
        start_time1 = time.time()
        coll_surf = p.map(particle, j)
        print ('orbit time: [{}]'.format((time.time() - start_time1)))
        p.close()
        p.join()
        coll_surf = [x for x in coll_surf if x != []]
        temp = np.load("/scratch/dsw310/sidm_data/surface/node1/coll.npy").tolist()
        temp.extend(np.concatenate(coll_surf).tolist())
        np.save("/scratch/dsw310/sidm_data/surface/node1/coll.npy",temp)


