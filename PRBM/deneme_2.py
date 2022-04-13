import numpy as np
import numpy.matlib as nummat
#%matplotlib widget
# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import style
from matplotlib.patches import Polygon
from scipy.optimize import fsolve
from cross_product import cross_product
from numpy import linalg as LA

# Geometrical and mechanical properties of the 2 sections TDCR

# Subsegment number and length
n = np.array([10,10]).reshape(2, 1); #number of spacerdisks on the 2 sections, n[1] corresponds to proximal section
n_disk= np.sum(n); #number of spacerdisks
L = np.array([200e-3,200e-3]).reshape(2, 1); # Lenght of the section in m
l=np.array([L[0]/n[0]*np.ones(n[0],dtype=int),L[1]/n[1]*np.ones(n[1],dtype=int)]).reshape(1,20); #array of segment lengths of each subsegment

# Tendon position on the disks
r_disk=10*1e-3; #distance of routing channels to backbone-center[m]
# Tendons actuating the proximal segment
r1=np.array([r_disk,0,0,1]).reshape(4,1); 
r2=np.array([0,r_disk,0,1]).reshape(4,1); 
r3=np.array([-r_disk,0,0,1]).reshape(4,1); 
r4=np.array([0,-r_disk,0,1]).reshape(4,1); 
# Tendons actuating the distal segment
r5=np.array([r_disk*np.cos(np.pi/4),r_disk*np.sin(np.pi/4),0,1]).reshape(4,1); 
r6=np.array([r_disk*np.cos(3*np.pi/4),r_disk*np.sin(3*np.pi/4),0,1]).reshape(4,1); 
r7=np.array([r_disk*np.cos(5*np.pi/4),r_disk*np.sin(5*np.pi/4),0,1]).reshape(4,1); 
r8=np.array([r_disk*np.cos(-1*np.pi/4),r_disk*np.sin(-1*np.pi/4),0,1]).reshape(4,1); 
p_tendon=np.concatenate((r1,r2,r3,r4,r5,r6,r7,r8),axis=1); #additional tendons can be added through additional columns


# Tendons actuating the proximal segment
r1=np.array([0,r_disk,0,1]).reshape(4,1); 
r2=np.array([r_disk*np.cos(-1*np.pi/6),r_disk*np.sin(-1*np.pi/6),0,1]).reshape(4,1); 
r3=np.array([r_disk*np.cos(7*np.pi/6),r_disk*np.sin(7*np.pi/6),0,1]).reshape(4,1); 
# Tendons actuating the distal segment
r4=np.array([0,r_disk,0,1]).reshape(4,1); 
r5=np.array([r_disk*np.cos(-1*np.pi/6),r_disk*np.sin(-1*np.pi/6),0,1]).reshape(4,1); 
r6=np.array([r_disk*np.cos(7*np.pi/6),r_disk*np.sin(7*np.pi/6),0,1]).reshape(4,1); 
p_tendon=np.concatenate((r1,r2,r3,r4,r5,r6),axis=1); #additional tendons can be added through additional columns

# Tendon tension
# F = [8 2 8 2 0 0 0 0];
F = np.array([8,0,0,0,0,0]).reshape(1,6);

# Backbone mechanical and geometrical properties
E=54*10**9;# Youngs modulus
nu=0.3; #Poissons ratio
G=E/(2*(1+nu)); #Shear modulus
ro=1.4/2*10**(-1*3); #outer radius of bb
ri=0; #inner radius of bb
I=(1/4)*np.pi*((ro**4)-(ri**4)); #moment of inertia
g=0; #9.81; #acceleration due to gravity
m_bb=0.0115*l*g; #mass of backbone #weight of the backbone expressed in kg/m multiplied by length and g
m_disk=0.2*1e-3*np.ones(n_disk,dtype=int)*g; #array of masses of each spacerdisk

# External tip forces and moments

Ftex = np.array([0,0,0,0]).reshape(4,1); # Force applied at the tip, expressed in global frame
Mtex = np.array([0,0,0,0]).reshape(4,1); # Moment applied at the tip

# Number of rigid bodies
nrb = 4;
# Length of rigid bodies, optimized with particle swarm algorithm in Chen 2011
gamma= np.array([0.125,0.35,0.388,0.136]).reshape(1,4)/np.sum(np.array([0.136,0.388,0.35,0.125]).reshape(1,4));

#phi0 = pi/2;
phi0 = 0;

# Initialization
rep = np.concatenate((np.array([phi0]),0*np.ones(nrb,dtype=int)),axis=0)
var0 = nummat.repmat(rep,1,n_disk);

###############################################################################
#function [T,Trb] = trans_mat_prbm(var,nrb,gamma,l,q,p)
def trans_mat_prbm(var,nrb,gamma,l,q,p):
    ##retruns rotation matrix from frame q to p
    T=np.eye(4);
    Trb = np.zeros((nrb*(q-p),4,4)); # trnasformation matrix from i-1 disk to Pj
    for iteration in range(p,q):
        theta = var[0][((nrb+1)*(iteration+1)-nrb-1):(nrb+1)*(iteration+1)-nrb+2];
        phi = var[0][((nrb+1)*(iteration+1)-nrb+2)];
        epsi = var[0][((nrb+1)*(iteration+1)-nrb+3)];
        
            
        R_phi = np.array([np.cos(phi),-1*np.sin(phi),0,0,
                              np.sin(phi),np.cos(phi),0,0,
                              0,0,1,0,
                              0,0,0,1]).reshape(4,4);
            
        R_phi_epsi = np.array([np.cos(epsi-phi),-1*np.sin(epsi-phi),0,0,
                          np.sin(epsi-phi),np.cos(epsi-phi),0,0
                          ,0,0,1,0,
                          0,0,0,1]).reshape(4,4);
            
        gamma_fun = np.array([1,0,0,0,0,1,0,0,0,0,1,gamma[0][0]*l[0][iteration],0,0,0,1]).reshape(4,4);
        Ti = np.matmul(R_phi,gamma_fun);
        
        Trb[nrb*((iteration+1)-(p+1)),:,:] = np.matmul(T,Ti);
        for k in range(0,nrb-1):
            temp = np.array([np.cos(theta[k]),0,np.sin(theta[k]),
                          gamma[0][k+1]*l[0][iteration]*np.sin(theta[k]),
                          0,1,0,0,
                          -1*np.sin(theta[k]),
                          0,np.cos(theta[k]),gamma[0][k+1]*l[0][iteration]*np.cos(theta[k]),
                          0, 0, 0, 1]).reshape(4,4);
            Ti = np.matmul(Ti,temp);
            #print(theta[k])
            #print("---------------------------")
            Trb[nrb*((iteration+1)-(p+1))+k+1,:,:] = np.matmul(T,Ti);
        #if q == 0 and p == 0:
            #T = np.eye(4);
        #else:
        T_temp = np.matmul(T,Ti);
        T = np.matmul(T_temp,R_phi_epsi)
    return T,Trb

def optim_f(var):
    res=np.zeros((n_disk*(nrb+1),1)); #nrb-1 revolute joints, 1 bending plane angle and 1 torsion angle
    F_prev = np.zeros((3,1));
    M_prev = np.zeros((3,1));
    
    for ss_i in range(n_disk-1,-1,-1): #iterating over each subsegment
        
        # Kinematics
        T_i,Trb = trans_mat_prbm(var,nrb,gamma,l,ss_i,ss_i-1); #returns transformation matrix from i to i-1
        theta=var[0][((nrb+1)*(ss_i+1)-nrb-1):(nrb+1)*(ss_i+1)-nrb+2];
        phi = var[0][((nrb+1)*(ss_i+1)-nrb+2)];
        ni = np.array([np.cos(phi+np.pi/2),np.sin(phi+np.pi/2),0]).reshape(3,1);
        epsi = var[0][((nrb+1)*(ss_i+1)-nrb+3)];
        
        p_ti=np.matmul(T_i,p_tendon); #position of tendon k at diski wrt i-1 frame
        p_i=np.matmul(T_i,np.array([0,0,0,1]).reshape(4,1)); #position of diski wrt i-1 frame
        norm_ct1= np.sqrt(np.sum(np.power((-1*p_ti[0:3,:]+p_tendon[0:3,:]),2), axis=0));

        Pi = np.zeros((4,nrb)); # position of the rigid bodies
        for k in range(0,nrb):
            Pi[:,k] = Trb[k,:,3]; # kontrol et Trb düzenini
        
        # Tendon tension on the different disks
        # Tension of the first set of tendons apply on the proximal segment
        # only
        nt = int(np.size(F)/np.size(n)); # Number of tendons per section
        temp_1 = np.concatenate((np.zeros((n[1][0],nt)), nummat.repmat(F[0][nt:],n[1][0],1)),axis=1);
        Fdisk = np.concatenate((nummat.repmat(F,n[0][0],1),temp_1),axis=0);
        
        # Direction orthogonal to the disk
        zi = T_i[:,2].reshape(4,1);
        if ss_i<n_disk:
            # Tendon force from disk ss_i to disk ss_i+1
            T_i2,Trb = trans_mat_prbm(var,nrb,gamma,l,ss_i+1,ss_i-1); #returns transformation matrix from i to i-1
            p_ti2=np.matmul(T_i2,p_tendon); #position of tendon k at diski wrt i-1 frame
            
            norm_ct2= np.sqrt(np.sum(np.power((-1*p_ti[0:3,:]+p_ti2[0:3,:]),2), axis=0));
            # Tendon force and moment: Eq (9)
            Fi = np.array([((p_tendon-p_ti)/nummat.repmat(norm_ct1,4,1))*nummat.repmat(Fdisk[ss_i,:],4,1)+((p_ti2-p_ti)/nummat.repmat(norm_ct2,4,1))*nummat.repmat(Fdisk[ss_i,:],4,1)]);
            if ss_i==n[0][0]:
                # Tip of segment 1
                # Consider the full force for tendon 1 to 3, remove
                # component orthogonal to the disk for tendon 4 to 6
                
                #Burda kaldım check et
                Fi = np.array([[Fi[:,0:3],
                                Fi[:,3:6]-nummat.repmat((zi*Fi[:,3:6]),[4,1])*nummat.repmat(zi,[1,3])]]).reshape(4,6);
            else:
                # Remove component orthogonal to the disk for tendon 1
                # to 6

                Fi = Fi - nummat.repmat(np.matmul(zi.T,Fi[0]),4,1)*nummat.repmat(zi,1,6);
        else:
            Fi = ((p_tendon-p_ti)/nummat.repmat(norm_ct1,4,1))*nummat.repmat(Fdisk[ss_i,:],4,1);
                    
        # Moment due to tendon force: Eq (12)
        Mi = cross_product(p_ti[0:3,:]-nummat.repmat(Pi[0:3,-1].reshape(3,1),1,np.size(F)),Fi[0:3,:][0]);
        
        # External forces and moments
        Rt, Trb_123 = trans_mat_prbm(var,nrb,gamma,l,ss_i-1,0);
        Fex, resid1,rank1,s11 = np.linalg.lstsq(Rt,Ftex);

        R_ex,Trb_1234 = trans_mat_prbm(var,nrb,gamma,l,n_disk,ss_i-1);
        
        p_ex = R_ex[0:3,3];
        qwe, resid2,rank2,s12 = np.linalg.lstsq(Rt[0:3,0:3],Mtex[0:3])
        Mex = qwe - cross_product((Pi[0:3,-1]-p_ex).reshape(3,1),Fex[0:3]);#+cross_product(p_i(1:3),Fex(1:3));      
        
        
        # Total forces and moments: Eq (17-18)
        if ss_i < n_disk:
            # Tip of segment 1
            Ftot = np.matmul(T_i[0:3,0:3],F_prev) + np.sum(Fi[0][0:3,:],axis=1).reshape(3,1);
            
            Mtot = np.matmul(T_i[0:3,0:3],M_prev) + cross_product((T_i2[0:3,3]-Pi[0:3,-1]).reshape(3,1),np.matmul(T_i[0:3,0:3],F_prev)) + np.sum(Mi,axis=1).reshape(3,1);
            
        else: 
            # Tip of segment 2
            Ftot =  np.sum(Fi[1:3,:],axis=1);
            Mtot = np.sum(Mi,axis=1);
        
        # Bending stiffness at each joint
        K = np.array([3.25*E*I/l[0][ss_i],2.84*E*I/l[0][ss_i],2.95*E*I/l[0][ss_i]]).reshape(1,3);

        for k in range(0,nrb-1):
            # Static equilibrium
            Rb = Trb[k+1,0:3,0:3];
            Mnetb = np.matmul(Rb.T,(cross_product((Pi[0:3,-1]-Pi[0:3,k]).reshape(3,1),Ftot[0:3]+Fex[0:3])+Mtot+Mex));
            res[(nrb+1)*(ss_i+1)-nrb+k-1] = K[0][k]*theta[k] - Mnetb[1];
        
        # Geometrical constraint for determining phi
        Mnet = cross_product(p_i[0:3],Ftot[0:3]+Fex[0:3])+Mtot+Mex;  # Net moment at disk i in frame i
        Mphi = Mnet; 
        Mphi[2]=0;
        res[(nrb+1)*(ss_i+1)-(nrb)+3] = np.matmul(ni.T,(Mphi))-LA.norm(Mphi);
        
        # Torsion
        Ri = T_i[0:3,0:3];
        Mepsi = np.matmul(Ri,Mnet);
        res[(nrb+1)*(ss_i+1)-(nrb)+3] = Mepsi[2]-2*G*I/l[0][ss_i]*epsi;
        
        F_prev = Ftot[0:3];
        M_prev = Mtot;
    return res

res = optim_f(var0)