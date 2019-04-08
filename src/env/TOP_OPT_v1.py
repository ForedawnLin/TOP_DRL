from __future__ import division
import math 
import numpy as np

### for calc compliance ###
from scipy.sparse import coo_matrix 
from scipy.sparse.linalg import spsolve


import sys
# print (sys.version)

class TOP_OPT_v1(): 
	### The first version (naive) environment for TOP_OPT_v1 
	### Caution: Reset the class when you first use it 
	
	def __init__(self): 
		### discretization settings ###
		self.nEleY=3 ### number of rows of the discretization of the object     
		self.nEleX=4 ### number of colns of ....
		self.nNodeX=self.nEleX+1 ## number of the nodes in X 
		self.nNodeY=self.nEleY+1 ## # of the nodes in Y
		
		### calc BC domain (coln major linear index) ### 
		self.BC_domain=set(np.arange(self.nNodeY)).union(set(np.arange((self.nNodeX-1)*self.nNodeY,self.nNodeX*self.nNodeY)).union( \
		set(np.arange(0,(self.nNodeX-1)*self.nNodeY+1,self.nNodeY)).union(set(np.arange(\
		self.nNodeY-1,self.nNodeX*self.nNodeY,self.nNodeY)))))
		self.node_domain=set(np.arange(self.nNodeX*self.nNodeY))
		# print (self.BC_domain)

		### agent state (need to reset first) ### 
		self.state=[]  ## init state 
		self.init_material=[] ### sum of initial material distribution 
		self.x_old=[] ### the material distribution from last update
	
		### material properties ###
		self.E=1    ### material young's modulus 
		self.nu=0.3 ### possion's ratio 

		### optimization setting ### 
		self.Emin=1e-9  ### min E for void element 
		self.Emax=1.0  ### max E for an element (usually same as self.E)
		self.vol_frac=0.4 ## shuold be a user input 
		self.penal=3 ### penalty for x to E relationship 
		self.th_terminate=1 ## Threshold for game(optimization) termination  

		### For complinace calculation (to comply with 88 lines) ###
		self.f=np.zeros((2*self.nNodeY*self.nNodeX,1)) ## init initial force condition (same as "f" in 88 line, e.g [0,1] is the first node, odd num force in y, even num is force in x) 
		self.u=np.zeros((2*self.nNodeY*self.nNodeX,1)) ## init displacement vector (same as 'u' in 88 line, the index has the same interpretation as f)
		self.free_BC_vec=[] ## init fixed BC (reset it first) (same as "free" in 88 line, e.g [0,1] is the first node, odd num free in y, even num is free in x) 
		


	def reset(self):
		### The function reset material distribution, the BC, init force, and the vol frac
		###  output: n*m*5 state matrix ( material_distri, BC, init_force*2(in x,y), vol_frac )
		### 1. reset material distri 
		material_distri=np.ones((self.nNodeY,self.nNodeX),dtype=int)  ### n*m, use nNode instead of nEleY to align with BC and init force matrix 
		self.x_old=np.ones((self.nEleY,self.nEleX),dtype=int)
		material_distri[:,0]=0 
		material_distri[0,:]=0  ### make the first row and coln zero since these are imaginary material 
		self.init_material=material_distri.sum()


		### 2. reset BC: fixed BC for now (config is the same as 88 line codes) #####
		x_fixed=[0,1,2,3]
		y_fixed=[19]
		BC_mat=self.__reset_BC__(x_fixed,y_fixed)  ### n*m
	

		### 3. reset force (same as 88 line): apply a force on boundary condition: 2 channels, first one is for x dir and the second one is for y dir 
		f_mat=np.zeros((self.nNodeY,self.nNodeX,2),dtype=float)   ### n*m*2
		fs=[[0,0,1,-1]]; ### initial forces, [x_pos,y_pos,direction(0:x;1:y),value]
		f_mat[fs[0][0],fs[0][1],fs[0][2]]=fs[0][3]  ## applying forace on the top left corner pointing downwards 
		### To comply with 88 lines (efficiency can be improved) ###
		f_index=[((ele[0]+ele[1]*self.nNodeY)*2)**(1-ele[2])*((ele[0]+ele[1]*self.nNodeY)*2+1)**(ele[2]) for ele in fs]
		f_val=[ele[3] for ele in fs] 
		self.f[f_index]=f_val 
		### 88 lines ends ###

		
		### 4. reset vol frac: constant across all elements 
		vol_frac=self.vol_frac
		vol_mat=vol_frac*np.ones((self.nNodeY,self.nNodeX),dtype=float)  ### n*m
	

		### concatenate all the chanvol_fracnels 
		self.state=np.dstack((material_distri,BC_mat,f_mat,vol_mat))  ### n*m*5
		return self.state  


	def step(self,action):
		### action: an nEleY*nEleX binary matrix (int). 1: flip that element 0: do not flip  
		
		### update states 
		x_new=self.state[1:self.nNodeY,1:self.nNodeX,0] ### material distribution x 
		x_new=(-x_new+action)*action+(-action+1)*x_new  ### flip operatin
		compliance=self.__calc_compliance__(x_new)
		self.state[1:self.nNodeY,1:self.nNodeX,0]=x_new ### update state  
		
		### calc rewards 
		vol_frac_new=x_new.sum()/self.init_material
		reward_vol=-compliance*abs(vol_frac_new-self.vol_frac)*100 ### 100 is a scale, when vol_frac is in 0.01 diff, reward is comparable with comlinace 
		reward_c=-compliance
		# print ('compliance',-compliance)
		reward=reward_vol+reward_c

		### termination  (0: not terminate 1: terminate) 
		is_terminate=0 
		if abs((self.x_old-x_new).sum())<=self.th_terminate:
			is_terminate=1
		self.x_old=x_new 

		return self.state,reward, is_terminate,{}


############## Below is helper functions (private) #######################
	def __reset_BC__(self,x_fixed,y_fixed):
		### This funciton defines the fixed node in BC for x and y direction
		### y_fixed: the node indices that are fixed in y directio; linear indice in row major  
		### 0:  not helpful region (inner region)  1: fixed in x  2:fixed in y  3:fixed in xy 4:free 
		set_y=set(y_fixed)
		set_x=set(x_fixed)
		assert self.BC_domain.intersection(set_x)==set_x and \
		self.BC_domain.intersection(set_y)==set_y,  "Boundary selection invalid " 

		#### find fixed x,y,xy ####  
		xy=set_x.intersection(set_y)
		y_only=set_y-xy
		x_only=set_x-xy 
		free_nodes=self.BC_domain-set_y-set_x

		### to comply with 88 lines ###
		free_nodes_x=np.array(list(self.node_domain-set_x))*2
		free_nodes_y=np.array(list(self.node_domain-set_y))*2+1 
		self.free_BC_vec=np.union1d(free_nodes_x,free_nodes_y)
		### 88 lines ends ####

		#### assign values for differen BC conditions for each element 
		Shape=(self.nNodeY,self.nNodeX)
		BC=np.zeros(Shape,dtype=int)
		if x_only!=set():
			BC[np.unravel_index(list(x_only),Shape,'F')]=1
		if y_only!=set():
			BC[np.unravel_index(list(y_only),Shape,'F')]=2
		if xy!=set():
			BC[np.unravel_index(list(xy),Shape,'F')]=3
		if free_nodes!=set():
			BC[np.unravel_index(list(free_nodes),Shape,'F')]=4

		return BC 


	def __calc_compliance__(self,x_new):
		### x: material distribution in nEleY*nEleX  
		### return: compliance, a scalar 
		Emin=self.Emin
		Emax=self.Emax
		nelx=self.nEleX
		nely=self.nEleY
		free=self.free_BC_vec
		penal=self.penal
		f=self.f
		u=self.u
		# dofs:
		ndof = 2*(nelx+1)*(nely+1)
		# Allocate design variables (as array), initialize and allocate sens.
		KE=self.__lk__()
		# x=self.vol_frac * x_new.reshape(nelx*nely,order='F') ### reshape for compliance calc (keep for reference)
		x=x_new.reshape(nelx*nely,order='F') ### reshape for compliance calc
		xPhys=x.copy()  ### may filter xPhy in the future ####
		

		edofMat=np.zeros((nelx*nely,8),dtype=int)
		for elx in range(nelx):
			for ely in range(nely):
				el = ely+elx*nely
				n1=(nely+1)*elx+ely
				n2=(nely+1)*(elx+1)+ely
				edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])

		# Construct the index pointers for the coo format
		iK = np.kron(edofMat,np.ones((8,1))).flatten()
		jK = np.kron(edofMat,np.ones((1,8))).flatten()  
		sK = ((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
		K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
		
		# Remove constrained dofs from matrix
		K = K[free,:][:,free]
		# Solve system 
		u[free,0]=spsolve(K,f[free,0])  
		# Objective and sensitivity
		ce = (np.dot(u[edofMat].reshape(nelx*nely,8),KE) * u[edofMat].reshape(nelx*nely,8) ).sum(1)
		obj=( (Emin+xPhys**penal*(Emax-Emin))*ce ).sum()
		return obj


	def __lk__(self):
		## calc element stiffness matrix 
		E=self.E
		nu=self.nu
		k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
		KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
		[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
		[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
		[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
		[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
		[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
		[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
		[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
		return (KE)


	def test(self): 
		self.reset() 
		action=np.zeros((self.nEleY,self.nEleX))
		# action[2,3]=1
		# action[1,1]=1
		self.state,reward, is_terminate,a=self.step(action)
		# print('state',self.state.shape)
		# print ('reward',reward)
		# print('terminate',is_terminate)
		# print('a',a)


if __name__=='__main__':
	Test=TOP_OPT_v1() 
	Test.test()

