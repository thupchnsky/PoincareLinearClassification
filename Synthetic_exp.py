#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.

import numpy as np
import matplotlib.pyplot as plt
from Perceptrons import *
import argparse
import os
import time
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import *
import warnings
import functools
warnings.filterwarnings("ignore")


"""
This is the code for our experiments on sythetic data.
Note that you can comment out the methods that you don't want to test in the Fun2parallel.
This can speed up the experiment.
"""

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--N', type=int, default=100000)
	parser.add_argument('--d', type=int, default=2)
	parser.add_argument('--gamma', type=float, default=0.01)
	parser.add_argument('--R', type=float, default=0.95)
	parser.add_argument('--a', type=float, default=0.)
	parser.add_argument('--thread', type=int, default=20)
	parser.add_argument('--chunksize', type=int, default=1)
	parser.add_argument('--Repeat', type=int, default=20)
	parser.add_argument('--savepath', type=str)
	args = parser.parse_args()

	if not os.path.exists(args.savepath):
		os.makedirs(args.savepath)

	N = args.N
	d = args.d
	R = args.R
	gamma = args.gamma
	a = args.a

	def Fun2parallel(Dummy=None,N=1000,gamma=0.1,d=2,p=None,w=None,R=0.95,a=0):
	#     Generate random seed!!!!
	    pid = mp.current_process()._identity[0]
	    np.random.seed(pid)
	    Record = np.zeros((3,5))

	#     for Rp in tqdm(range(Repeat)):
	    Flag = False
	    Gen_count = 0
	    p_input = p
	    w_input = w
	    while not Flag:
	        Gen_count += 1
	        if Gen_count > 100:
	            return Record
	        N,X,y,w,p = Poincare_Uniform_Data(N,d,gamma,R = R, p = p_input,w=w_input)
	        if (len(y[y==1])>10) and (len(y[y==-1])>10):
	            Flag = True
	        

	    y.astype('int32')

	# # Enable this part is you want to use estimated p.
	# # However, you can't use Graham scan for the case d>2. You have to use other heuristics.
	# #     Time Graham scan first
	#     start = time.time()
	#     CH1 = ConvexHull(X[:,y==1])
	#     CH2 = ConvexHull(X[:,y==-1])
	#     MDpair = minDpair(CH1,CH2)
	#     p_hat = Weightedmidpt(MDpair[:,0],MDpair[:,1],0.5)
	#     tGraham = time.time()-start

	#    Time P2L
	    start = time.time()
	    DUMMY = P2L(X)
	    tP2L = time.time()-start

	#     our_perceptron part
	    start = time.time()
	    w1,Record[2,0] = HP(X,y,p,gamma,R,a=None,option='first')
	    Record[1,0] = time.time()-start#+tGraham
	    Record[0,0],_ = Eval(X=X,y=y,p=p,w1 = w1)


	# #     our_SOP
	    start = time.time()
	    xi, C,Record[2,1] = HP(X,y,p,gamma,R,a=a,option='second')
	    Record[1,1] = time.time()-start#+tGraham
	    Record[0,1] = Eval(X=X,y=y,p=p,xi=xi, C=C)

	#     Translate y from {-1,1} to {0,1}
	    y_SVM = np.round((y+1)/2).astype('int32')


	#     our_SVM
	    Acc, Record[1,2], w_Psvm = tangent_hsvm(X.T, y_SVM, X.T, y_SVM, 1000,p=p)
	    Record[0,2] = Acc*100

	#     cho_SVM
	    Acc, ChoTime = cho_hsvm(X.T, y_SVM, X.T, y_SVM, 10)
	    Record[0,3] = Acc*100
	    Record[1,3] = ChoTime - tP2L

	#     Euclidean_SVM
	    Acc, Record[1,4], w_eu = euclidean_svm(X.T, y_SVM, X.T, y_SVM, 1000)
	    Record[0,4] = Acc*100
	    
	    return Record    

	mypool = Pool(args.thread)
	chunksize = args.chunksize
	Repeat = args.Repeat
	# First axis: Acc, Mistakes,time
	# Second axis: method [our_perceptron,our_SOP, our_SVM, cho_SVM, Euclidean_SVM]
	Record_meta = np.zeros((3,Repeat))
	# pbar = tqdm(total=Repeat)
	
	
	for alpha in [1,2,3]:
	    Rp = alpha*R/5
	    Record_meta = np.zeros((3,5,Repeat))
#         pbar = tqdm(total=Repeat)
#         theta = np.random.random()
	    p = np.random.normal(size=d)
	    p = p/np.linalg.norm(p)*Rp
#         w = p/Rp
	    fn = functools.partial(Fun2parallel,N = N,gamma = gamma,d=d,p=p,w=None,R=R,a=a)
	    print('N,gamma,alpha,d:'+str(N)+', '+str(gamma)+', '+str(alpha)+', '+str(d)+' Start!')
        
	    for ind, res in enumerate(mypool.imap(fn, range(Repeat)), chunksize):
	        Record_meta[:,:,ind-1] = res
#             pbar.update()
	    np.save(args.savepath+'/Result_N_'+str(N)+'_margin_'+str(gamma)+'_alpha_'+str(alpha)+'_d_'+str(d),Record_meta)
	    print('N,gamma,alpha,d:'+str(N)+', '+str(gamma)+', '+str(alpha)+', '+str(d)+' Done!')

	        