# Glossary of terms introduced throughout the course

(constantly updated)

- features = observables = 'branches' (in ROOT terminology) = 'columns' of table
- in supervised learning x_i - vector of features, y_i is target (something we want to reconstruct).
  - optionally we may have sample weights w_i = 'cost of mistaking' at predicting this sample
- most frequent problems in supervised learning: classification and regression
- in unsupervised learning we are left only with vectors x_i
- knn - k Nearest neighobours
- ρ (rho) denotes distance in the space of features
- p denotes probability, p_{+1}(x) = p(y = +1 | x) is a probability of x belonging to the class 1
- < a, b > is dot product
- < f(z) >_{over some set of z} is taking average of f(z) over the specicied set of z
- η (eta) is learning rate a.k.a shrinkage (not to be messed up with pseudorapidity!)


# Things specific to this course

- for binary clasification target is taken to be y_i = 1 (signal) or y_i = - 1 (background)
- classifiers / regressors have intermediate step - a decision function, which is denoted as d(x) for simple models.
  - Example: d(x) = < w, x >
- for ensembles D(x) is used as notion. 
  - Example: D(x) = \sum_j d_j(x) - simply summing decisions of weak learners
- Beautiful L is *loss*, something we are minimizing in the algorithm. Typically, this is (upper) estimate of our risks, 
  taken to have nice optimization properties.
- Regularizations are a way to effectively bound the combinations checked during optimization
  - typically we add L_1, L2 or mixed regularization.
  - (those are very nice)
  - used to prevent overfitting (see below)
- w (if vector), W (if matrix) are parameters of ML models to be optimized (see previous point).
  - called parameters of a model or weights of a model. 
  - conflict of demotions: w_i are sample weights, because i is indexing samples in the data   

# Some general things


- Process of using ML is split into
  - training = fitting = learning
  - and predicting or transforming
- *cross-validation* is a process of getting reliable estimation of quality
  - in simplest case, we estimate the quality on a separate holdout - part of the data not used in training.   
- *linear models* are using linear decision function d(x) = < w, x >
- *generalized* linear models, e.g. SVM, are using Kernel functions (which are dot products in some space)
- Decision tree operates by checking `splits`, e.g. mass > 3.7
  - split is described by feature used (mass) and threshold (3.7)
- Decision tree has pre-stopping conditions and can be post-pruned (simplified) after it was trained
- *subsampling* and *bagging* are the same as subsampling with / without replacement
- RSM subsamples features
- Bagging is used for taking subsets of samples in RandomForest
- *overfitting* - a very vague term in ML to denote problems with trained formula. 
  - see lectures for more details
  - if you use term overfitting, you'd better directly explain what do you mean by this
- Gradient Boosting - general technique, but typically we run it over decision trees
  - GB = GBDT = GBRT = GBM ~= MART are all typically used as names for gradient boosting over decision trees

