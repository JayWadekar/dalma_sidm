#!/usr/bin/env python
#Constant cross-section

#SBATCH -p parallel
#SBATCH -o "/scratch/dsw310/sidm_data/surface/node2/output.log"
#SBATCH -e "/scratch/dsw310/sidm_data/surface/node2/error.log"
#SBATCH -n 45
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=dsw310
#SBATCH --time=12:00:00

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

from galpy.potential import MiyamotoNagaiPotential, NFWPotential, PowerSphericalPotentialwCutoff, MWPotential2014, vcirc, vesc
from galpy.orbit import Orbit
from galpy.util import bovy_conversion

#mp= MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=.6)
nfw= NFWPotential(a=16/8.,normalize=.35)
#bp= PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=0.05)
mwp = MWPotential2014

column_convert = 1673./800*1.24/6.04
t_tot = 1./bovy_conversion.time_in_Gyr(220.,8.); n_t = 900.
ts= np.linspace(0,t_tot,n_t)
sigmav_cmz=10./220. 
sigmav = 7./220.
sigmav_halo = 50./220.

tfrac=np.arange(0.015,0.31,0.015)
for i in range(2,11,2): tfrac = np.append(tfrac,np.arange(i-0.15,i+0.16,0.015))    
tfrac = np.append(tfrac[:-1],[10.15,11.15])#last time not counted but choose high enough value
lines = np.load('/scratch/dsw310/sidm_source/Input_data/equilibrated_MW_23.npy')
cst_b = 1.989*220.*3.154/pow(3.086,3)*bovy_conversion.time_in_Gyr(220.,8.)*t_tot/n_t
cst_Ghalo = 0.22*3.154*1.67*1.35*pow(5,-1.5)*bovy_conversion.time_in_Gyr(220.,8.)*t_tot/n_t

#Constructing the gas distributions
gas_dens_data = np.loadtxt('/scratch/dsw310/sidm_source/Input_data/hI_dens_galpy.dat') #M_sun/pc^3
gas_dens_log = interp1d(gas_dens_data[:,0], np.log(gas_dens_data[:,1]),fill_value='extrapolate')
gas_scale_data = np.loadtxt('/scratch/dsw310/sidm_source/Input_data/hI_height_galpy.dat')#kpc
gas_scale_log = interp1d(gas_scale_data[:,0], np.log(gas_scale_data[:,1]),fill_value='extrapolate')
  
def Coll(R,phi,z,vR,vT,vz):
    theta = np.arccos(2.*random.random() - 1.)
    phi = 2.*np.pi*random.random()
    veldm = np.array([vR,vT,vz])
    He_flag=False
    if (abs(z)>(2./8.)):
        velb = np.array([np.random.normal(0,sigmav_halo), 183./220.+np.random.normal(0,sigmav_halo), np.random.normal(0,sigmav_halo)])
    elif (R<(0.5/8.)):
        velb = np.array([np.random.normal(0,sigmav_cmz), vcirc(mwp,R)+np.random.normal(0,sigmav_cmz), np.random.normal(0,sigmav_cmz)])
    else:
        if(random.random()>0.15231788):
            He_flag=True;velb = np.array([np.random.normal(0,sigmav/2.), vcirc(mwp,R)+np.random.normal(0,sigmav/2.), np.random.normal(0,sigmav/2.)])
        else:   velb = np.array([np.random.normal(0,sigmav), vcirc(mwp,R)+np.random.normal(0,sigmav), np.random.normal(0,sigmav)])
    if (He_flag==False):
        vel = 1./3.*velb+2./3.*veldm-1./3.*np.linalg.norm(veldm-velb)*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        velb_final = velb+2.*veldm-2.*vel
        energy=np.linalg.norm(velb_final)**2-np.linalg.norm(velb)**2
    else:
        vel = 2./3.*velb+1./3.*veldm-2./3.*np.linalg.norm(veldm-velb)*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        velb_final = 2.*velb+veldm-vel;He_flag=False
        energy=4.*(np.linalg.norm(velb_final)**2-np.linalg.norm(velb)**2)
    return [vel,velb_final,energy]
    

def particle(num):
    global save_surf
    o= Orbit(vxvv=lines[num]) #Initial condition [R,vR,vT,z,vz,phi] 
    #print "Num,R,z:", ([num,lines[num][0]*8., lines[num][3]*8.])
    column_b=0.; column0_b = -column_convert*np.log(random.random())
    time = 0.; tpick = 0.
    numcoll_b = 0; i_tf = 0
    surf=[[o.R(tpick)*8.,o.phi(tpick),o.z(tpick)*8.,o.vR(tpick)*220.,o.vT(tpick)*220.,o.vz(tpick)*220.]]
    coll_save=[]
    flag=True
    
    #See if the orbit should be excluded--------------------------
    ts_use = np.linspace(0,t_tot*10.15,n_t*8.) #sampling arbitrarily chosen 
    cst_b_use = cst_b*10.15/8.; cst_Ghalo_use = cst_Ghalo*10.15/8.
    
    # Orbit integration-----------------------------------------
    while (time<10.15):
        o.integrate(ts_use,MWPotential2014,method='dopr54_c')
        for t in ts_use:
            column_b+= (np.exp(gas_dens_log(o.R(t)*8.)-pow(o.z(t)*8.,2)/np.exp(gas_scale_log(o.R(t)*8)))*np.linalg.norm(np.array([o.vR(t),o.vT(t)-vcirc(mwp,o.R(t)),o.vz(t)]))*cst_b_use) + (pow(1+(o.r(t)/(5./8.))**2,-0.75)*np.linalg.norm(np.array([o.vR(t),o.vT(t)-183/220.,o.vz(t)]))*cst_Ghalo_use) 
                    
            if (column_b > column0_b):
                numcoll_b+=1; column_b = 0.; column0_b = -column_convert*np.log(random.random())  
                [vel,velb_final,energy]=Coll(o.R(t),o.phi(t),o.z(t),o.vR(t),o.vT(t),o.vz(t))
                time+=t*bovy_conversion.time_in_Gyr(220.,8.)
                coll_save.append([o.R(t)*8.,o.phi(t),o.z(t)*8.,velb_final[0]*220.,velb_final[1]*220.,velb_final[2]*220.,energy*(220.**2),num,time])
                flag=False
                
            elif (t==ts_use[-1]): #counts for no collision in the given time  
                vel=[o.vR(t),o.vT(t),o.vz(t)]
                time+=t*bovy_conversion.time_in_Gyr(220.,8.)
                flag=False
                
            if(flag==False):
                while (time>=tfrac[i_tf]):
                    tpick = t- (time-tfrac[i_tf])/bovy_conversion.time_in_Gyr(220.,8.)
                    surf.append([o.R(tpick)*8.,o.phi(tpick),o.z(tpick)*8.,o.vR(tpick)*220.,o.vT(tpick)*220.,o.vz(tpick)*220.])
                    i_tf+=1
                o= Orbit(vxvv=[o.R(t),vel[0],vel[1],o.z(t),vel[2],o.phi(t)]) 
                flag=True   
                break
        ts_use=ts; cst_b_use= cst_b; cst_Ghalo_use= cst_Ghalo
        if (numcoll_b>40 and np.linalg.norm([coll_save[-1][0]-coll_save[-2][0],coll_save[-1][2]-coll_save[-2][2]])<0.1): #More than 40 and close collisions
            while(len(surf)<len(tfrac)):
                surf.append([coll_save[-1][0],coll_save[-1][1],coll_save[-1][2],coll_save[-1][3],coll_save[-1][4],coll_save[-1][5]])
            break   
              
    if (len(surf)!=len(tfrac)):
        lock.acquire()
        f_log = open('log_sidm_surf.txt', 'a')
        np.savetxt(f_log,np.array(surf))
        f_log.write('Num,R,z,tpick: [{},{},{},{}] \n \n'.format(num,lines[num][0]*8.,lines[num][3]*8.,tpick))
        f_log.close()
        lock.release()
        surf = []
        while (len(surf)<len(tfrac)):
            surf.append([o.R(0)*8.,o.phi(0),o.z(0)*8.,o.vR(0)*220.,o.vT(0)*220.,o.vz(0)*220.])                    
    return [surf,coll_save]
    
def init(l):
    global lock
    lock = l

#deleting prev output files
#open('log_sidm_surf.txt', 'w').close()
if (os.path.exists("/scratch/dsw310/sidm_data/surface/node2/coll.npy")==False):
    np.save("/scratch/dsw310/sidm_data/surface/node2/coll.npy",[])
for i in range (0,len(tfrac)):
    if (os.path.exists("/scratch/dsw310/sidm_data/surface/node2/"+str(i)+".npy")==False):
        np.save("/scratch/dsw310/sidm_data/surface/node2/"+str(i)+".npy",[])

if __name__ == '__main__':
    for k in range(19,20): #started from 14 #end at 20
        j=np.arange(10000*k,10000*(k+1),1)
        l = multiprocessing.Lock()
        p = multiprocessing.Pool(initializer=init,initargs=(l,),processes=28) 
        start_time1 = time.time()
        save_surf = p.map(particle, j)
        print ('orbit time: [{}]'.format((time.time() - start_time1)))
        p.close()
        p.join()
        #start_time1 = time.time()
        snaps=[]; coll=[]
        for l in save_surf:
            snaps.append(l[0]); coll.append(l[1])
        snaps = np.hstack(snaps)
        coll = [x for x in coll if x != []]
        temp = np.load("/scratch/dsw310/sidm_data/surface/node2/coll.npy").tolist()
        temp.extend(np.concatenate(coll).tolist())
        np.save("/scratch/dsw310/sidm_data/surface/node2/coll.npy",temp)
        for i in range (0,len(tfrac)):
            temp = np.load("/scratch/dsw310/sidm_data/surface/node2/"+str(i)+".npy").tolist()
            temp.extend(np.concatenate([np.array_split(snaps[i],len(j))]).tolist())
            np.save("/scratch/dsw310/sidm_data/surface/node2/"+str(i)+".npy",temp)
        #print ('file time: [{}]'.format((time.time() - start_time1)))



