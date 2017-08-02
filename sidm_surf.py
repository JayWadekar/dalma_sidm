#!/usr/bin/env python

#SBATCH -p serial

#SBATCH -o "/scratch/dsw310/sidm_data/surface/node1/output.log"
#SBATCH -e "/scratch/dsw310/sidm_data/surface/node1/error.log"
#SBATCH -n 28
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=dsw310
#SBATCH --time=00:10:00

import os
import numpy as np
import matplotlib.pyplot as plt
import galpy
from pylab import *
import random
import math
import multiprocessing
import time
import scipy
from scipy.interpolate import interp1d

sys.path.append(os.getcwd())

from galpy.potential import MiyamotoNagaiPotential, NFWPotential, PowerSphericalPotentialwCutoff, MWPotential2014, vcirc
from galpy.orbit import Orbit
from galpy.util import bovy_conversion

#mp= MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=.6)
nfw= NFWPotential(a=16/8.,normalize=.35)
#bp= PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=0.05)
mwp = MWPotential2014

column_ref_b = 1670/500*1.24/6.04
t_tot = 1./bovy_conversion.time_in_Gyr(220.,8.); n_t = 900.
ts= np.linspace(0,t_tot,n_t) 
sigmav = 8/np.sqrt(3)/220.


tfrac = np.arange(0.05,10.5,5); tfrac = np.append(tfrac,[10.5,11.5])#last time not counted but choose high enough value
lines = np.load('equilibrated_MW.npy')
cst_b = 1.989*220.*3.154/pow(3.086,3)*bovy_conversion.time_in_Gyr(220.,8.)*t_tot/n_t
cst_Ghalo = 0.22*3.154*1.67*1.35*pow(5,-1.5)*bovy_conversion.time_in_Gyr(220.,8.)*t_tot/n_t

#Constructing the gas distributions
gas_dens_data = np.loadtxt('hI_dens_galpy.dat') #M_sun/pc^3
gas_dens_log = interp1d(gas_dens_data[:,0], np.log(gas_dens_data[:,1]),fill_value='extrapolate')
gas_scale_data = np.loadtxt('hI_height_galpy.dat')#kpc
gas_scale_log = interp1d(gas_scale_data[:,0], np.log(gas_scale_data[:,1]),fill_value='extrapolate')

def Coll(R,phi,z,vR,vT,vz,r):
    theta = np.arccos(2.*random.random() - 1.)
    phi = 2*np.pi*random.random()
    veldm = np.array([vR,vT,vz])
    if (abs(z)>(2/8.) and r>(3/8.)):
        velb = np.array([np.random.normal(0,sigmav)*math.exp(-r), 183/220.+np.random.normal(0,sigmav)*math.exp(-r), np.random.normal(0,sigmav)*math.exp(-r)])
    else:
        velb = np.array([np.random.normal(0,sigmav)*math.exp(-R), vcirc(mwp,R)+np.random.normal(0,sigmav)*math.exp(-R), np.random.normal(0,sigmav)*math.exp(-R)])
    vel = 1./3.*velb+2./3.*veldm-1./3.*np.linalg.norm(veldm-velb)*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    return (vel)
    

def particle(num):
    global save_surf
    o= Orbit(vxvv=lines[num]) #Initial condition [R,vR,vT,z,vz,phi] 
    #print "Num,R,z:", ([num,lines[num][0]*8., lines[num][3]*8.])
    column_b=0.; column0_b = -column_ref_b*np.log(random.random())
    time = 0.; tpick = 0.
    numcoll_b = 0; i_tf = 0
    surf=[[o.R(tpick)*8.,o.phi(tpick),o.z(tpick)*8.,o.vR(tpick)*220.,o.vT(tpick)*220.,o.vz(tpick)*220.]]
    db_tpick = 0.
    flag=True
    
    #See if the orbit should be excluded--------------------------
    ts_use = np.linspace(0,t_tot*10.5,n_t*8) #co-efficients arbitrarily chosen 
    cst_b_use = cst_b*10.5/8.; cst_Ghalo_use = cst_Ghalo*10.5/8.
    
    # Orbit integration-----------------------------------------
    while (time<10.5):
        o.integrate(ts_use,MWPotential2014,method='dopr54_c')
        for t in ts_use:
            column_b+= (np.exp(gas_dens_log(o.R(t)*8.)-pow(o.z(t)*8.,2)/np.exp(gas_scale_log(o.z(t)*8)))*np.linalg.norm(np.array([o.vR(t),o.vT(t)-vcirc(mwp,o.R(t)),o.vz(t)]))*cst_b_use) + (pow(1+(o.r(t)/(5./8.))**2,-0.75)*np.linalg.norm(np.array([o.vR(t),o.vT(t)-183/220.,o.vz(t)]))*cst_Ghalo_use) 
                    
            if (column_b > column0_b):
                numcoll_b+=1; column_b = 0.; column0_b = -column_ref_b*np.log(random.random())  
                vel=Coll(o.R(t),o.phi(t),o.z(t),o.vR(t),o.vT(t),o.vz(t),o.r(t))
                flag=False
                
            elif (t==ts_use[-1]): #counts for no collision in the given time  
                vel=[o.vR(t),o.vT(t),o.vz(t)]
                flag=False
                
            if(flag==False):
                time+=t*bovy_conversion.time_in_Gyr(220.,8.)
                while (time>=tfrac[i_tf]):
                    tpick = t- (time-tfrac[i_tf])/bovy_conversion.time_in_Gyr(220.,8.)
                    surf.append([o.R(tpick)*8.,o.phi(tpick),o.z(tpick)*8.,o.vR(tpick)*220.,o.vT(tpick)*220.,o.vz(tpick)*220.])
                    i_tf+=1
                o= Orbit(vxvv=[o.R(t),vel[0],vel[1],o.z(t),vel[2],o.phi(t)]) 
                flag=True   
                break
        ts_use=ts; cst_b_use= cst_b; cst_Ghalo_use= cst_Ghalo
        if (numcoll_b>40):
            while(len(surf)<len(tfrac)):
                surf.append([o.R(0)*8.,o.phi(0),o.z(0)*8.,o.vR(0)*220.,o.vT(0)*220.,o.vz(0)*220.])
            break   
                
    if (len(surf)!=len(tfrac) or db_tpick!=0):
        lock.acquire()
        f_log = open('log_sidm_surf.txt', 'a')
        np.savetxt(f_log,np.array(surf))
        f_log.write('Num,R,z,tpick: [{},{},{},{}] \n \n'.format(num,lines[num][0]*8.,lines[num][3]*8.,db_tpick))
        f_log.close()
        lock.release()
        if (len(surf)!=len(tfrac)):
            surf = []
            while (len(surf)<len(tfrac)):
                surf.append([o.R(0)*8.,o.phi(0),o.z(0)*8.,o.vR(0)*220.,o.vT(0)*220.,o.vz(0)*220.])                    
    return(surf)
    
def init(l):
    global lock
    lock = l
    
print (particle(300))
'''
#deleting prev output files
#open('log_sidm_surf.txt', 'w').close()
for i in range (0,len(tfrac)):
    if (os.path.exists("/scratch/dsw310/sidm_data/surface/node1/"+str(i)+".npy")==False):
        np.save("/scratch/dsw310/sidm_data/surface/node1/"+str(i)+".npy",[])

if __name__ == '__main__':
    for k in range(4,5):
        j=np.arange((20*k),(20*(k+1)),1)
        l = multiprocessing.Lock()
        p = multiprocessing.Pool(initializer=init,initargs=(l,),processes=28) 
        start_time1 = time.time()
        save_surf = p.map(particle, j)
        print ('orbit time: [{}]'.format((time.time() - start_time1)))
        p.close()
        p.join()
        start_time1 = time.time()
        save_surf = np.hstack(save_surf)
        for i in range (0,len(tfrac)):
            temp = np.load("/scratch/dsw310/sidm_data/surface/node1/"+str(i)+".npy").tolist()
            temp.extend(np.concatenate([np.array_split(save_surf[i],len(j))]).tolist())
            np.save("/scratch/dsw310/sidm_data/surface/node1/"+str(i)+".npy",temp)
        print ('file time: [{}]'.format((time.time() - start_time1)))
'''


''' Junk code
                    if (tpick>t or tpick<0):
                        db_tpick = tpick
                        #print ('tpick wtf: {}\n'.format(tpick))
                        tpick = 0.
                        
                        
Collision saving code
rvb_final = np.append([o.R(t)*8.,o.phi(t),o.z(t)*8.],(velb+2.*veldm-2*vel)*220.)
 coll_rv.append(rvb_final.tolist()) 
'''
