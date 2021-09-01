# Official implementation of efficient linear classification in hyperbolic space

### [Highly Scalable and Provably Accurate Classification in Poincare Balls](paper link)

Programming language: Python 3.7. Tested on operating systems: Windows 10, CentOS 7.7.1908

### TBA

The Jupyter notebook `HP_single_exp.ipynb` contains a simple example code for our hyperbolic perceptron algorithms, hyperbolic perceptron from Weber et al. and Euclidean perceptron on synthetic data with visualization.

# Reproducing our experiments of Fig 7 and Table 1
```
python Synthetic_exp.py --savepath [your saving path] 
```
The experimental setting that can be changed are listed as follows: \
--N: Number of points (default: 100000) \
--d: Dimension (default: 2) \
--gamma: Margin (default: 0.01) \
--R: Upper bound of the norm of data points (default: 0.95) \
--a: The hyperparameter in the second order perceptron (default: 0) \
--thread: Number of threads used for parallelization (default: 20) \
--chucksize: Chucksize for parallelization (default: 1) \
--Repeat: Number of repeat of experiments (default: 20) 

Note that you can comment out some methods that you don't want to test in the file `Synthetic_exp.py`. We have a more detail instruction in it. 

The output will be saved as a (3,5,Repeat) numpy arrany. \
First axis: acc, mistakes (for perceptron only), running time. \
Second axis: methods. They are our hyperbolic perceptron, our second order hyperbolic perceptron, our hyperbolic SVM, SVM from Cho et al., Euclidean SVM.

## Contact
Please contact Chao Pan (chaopan2@illinois.edu), Eli Chien (ichien3@illinois.edu) if you have any question.
