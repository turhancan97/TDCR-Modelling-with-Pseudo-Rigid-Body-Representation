from cross_product import cross_product
import numpy.matlib as nummat

# function [var,exitflag,res] = prbm_solver(n,nrb,gamma,l,F,p_tendon,m_disk,m_bb,E,I,G,Ftex,Mtex,var0)

def prbm_solver(n,nrb,gamma,l,F,p_tendon,m_disk,m_bb,E,I,G,Ftex,Mtex,var0):
    ##returns the solved values for [beta, gamma, epsi] for each subsegment i
    n_disk = np.sum(n);
    #options1 = optimset('Display','iter','TolFun',1e-6,'MaxFunEvals',1500,'TolX',1e-6,'Algorithm','trust-region-dogleg'); 

    #tic
    #var,infodict = fsolve(optim_f,var0);
    #toc

## solver

    #function [res] = optim_f(var)
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
            ## norm bak
            norm_ct1=sqrt(sum((-p_ti(1:3,:)+p_tendon(1:3,:)).^2));
            Pi = np.zeros((4,nrb)); # position of the rigid bodies
            for k in range(0,nrb)
                Pi[:,k] = Trb[0][k,:,4]; # kontrol et Trb düzenini
            
            # Tendon tension on the different disks
            # Tension of the first set of tendons apply on the proximal segment
            # only
            nt = np.size(F)/np.size(n); # Number of tendons per section
            Fdisk = np.array([nummat.repmat(F,n(1),1).reshape(10,6),
                             np.array([zeros(n(2),nt).reshape(10,3),
                                       nummat.repmat(F(nt+1:end),n(2),1)]).reshape(10,6)]).reshape(20,6);
            
            # Direction orthogonal to the disk
            zi = T_i[0][end,:,3];
            
            if ss_i<n_disk:
                # Tendon force from disk ss_i to disk ss_i+1
                T_i2,Trb = trans_mat_prbm(var,nrb,gamma,l,ss_i+1,ss_i-1); #returns transformation matrix from i to i-1
                p_ti2=np.matmul(T_i2,p_tendon); #position of tendon k at diski wrt i-1 frame
                #norm bak
                norm_ct2=sqrt(sum((-p_ti(1:3,:)+p_ti2(1:3,:)).^2));
                
                # Tendon force and moment: Eq (9)
                Fi = np.array([((p_tendon-p_ti)/nummat.repmat(norm_ct1,4,1))*nummat.repmat(Fdisk[ss_i,:],4,1)+((p_ti2-p_ti)/nummat.repmat(norm_ct2,4,1))*nummat.repmat(Fdisk[ss_i+1,:],4,1)]);
                
                if ss_i==n[0]:
                    # Tip of segment 1
                    # Consider the full force for tendon 1 to 3, remove
                    # component orthogonal to the disk for tendon 4 to 6
                    Fi = np.array([[Fi[:,0:2],
                                    Fi[:,3:5]-nummat.repmat((zi*Fi(:,3:5)),[4,1])*nummat.repmat(zi,[1,3])]]).reshape(4,6);
                else:
                    # Remove component orthogonal to the disk for tendon 1
                    # to 6
                    Fi = Fi - nummat.repmat(zi*Fi,[4,1])*nummat.repmat(zi,[1,6]);
            else:
                Fi = ((p_tendon-p_ti)/nummat.repmat(norm_ct1,4,1))*nummat.repmat(Fdisk(ss_i,:),4,1);
                        
            # Moment due to tendon force: Eq (12)
            Mi = cross_product(p_ti[1:3,:]-nummat.repmat(Pi[0:2,end],1,np.size(F)),Fi[0:2,:]);
            
            # External forces and moments
            Rt, Trb_123 = trans_mat_prbm(var,nrb,gamma,l,ss_i-1,0);
            Fex = Rt\Ftex; 

            R_ex = trans_mat_prbm(var,nrb,gamma,l,n_disk,ss_i-1);
            p_ex = R_ex[0:2,3];
            qwe, resid,rank,s1 = np.linalg.lstsq(Rt[0:2,0:2],Mtex[0:2])
            Mex = qwe - cross_product(Pi[0:2,end]-p_ex,Fex[0:2]);#+cross_product(p_i(1:3),Fex(1:3));      
            
            
            # Total forces and moments: Eq (17-18)
            if ss_i < n_disk:
                # Tip of segment 1
                Ftot = np.matmul(T_i[0:2,0:2],F_prev) + np.sum(Fi[1:3,:],1);
                Mtot = np.matmul(T_i[0:2,0:2],M_prev) + cross_product(T_i2[0:2,3]-Pi[0:2,end],np.matmul(T_i[0:2,0:2]*F_prev)) + np.sum(Mi,axis=1);
            else 
                # Tip of segment 2
                Ftot =  np.sum(Fi[1:3,:],axis=1);
                Mtot = np.sum(Mi,axis=1);
            
            # Bending stiffness at each joint
            K = np.array([3.25*E*I/l[ss_i],2.84*E*I/l[ss_i],2.95*E*I/l[ss_i]]).reshape(1,3);

            for k in range(0,nrb-1)
                # Static equilibrium
                Rb = Trb[k+1,0:2,0:2];
                Mnetb = np.matmul(Rb,(cross_product(Pi[1:3],end)-Pi[0:2,k],Ftot[0:2]+Fex[0:2])+Mtot+Mex));
                res[(nrb+1)*(ss_i+1)-nrb+k-1]= K[k]*theta[k] - Mnetb[1];
            
            # Geometrical constraint for determining phi
            Mnet = cross_product(p_i[0:2],Ftot[0:2]+Fex[0:2])+Mtot+Mex;  # Net moment at disk i in frame i
            Mphi = Mnet; 
            Mphi[2]=0;
            #norm çöz
            res[(nrb+1)*ss_i-(nrb)+3] = np.matmul(ni,(Mphi))-norm(Mphi);
            
            # Torsion
            Ri = T_i[0:2,0:2];
            Mepsi = np.matmul(Ri,Mnet);
            res[(nrb+1)*ss_i-(nrb)+4] = Mepsi[2]-2*G*I/l[ss_i]*epsi;
            
            F_prev = Ftot[0:2];
            M_prev = Mtot;