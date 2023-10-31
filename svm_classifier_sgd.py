import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

class svm_primal_cls:
  """ Class implementing the primal SVM algorithm:
  "Stochastic gradient descent algorithm for soft-margin SVM"
  """

  def __init__(
    self,
    C:int = 1000, 
    eta:float = 0.1, 
    xlambda:float = 0.01, 
    nitermax:int = 10
  ):
    self.C = C                # Penalty coefficient
    self.eta = eta            # Stepsize
    self.nitermax = nitermax  # Number of iterations
    self.xlambda = xlambda    # Penalty parameter 1/C
    self.w = None             # Weights
    
    return

  ## ---------------------------------------------------------
  def fit(self,X:np.ndarray,y:np.ndarray):
    """ Train the support Vector Machine using Stochastic gradient descent algorithm

    Parameters:
    -----------
      X: 2d array of input data.
      y: 1d array of target variable data (+1, -1 labels)
    
    Modifies:
    --------- 
      self.w:  Weight vector         
    """

    m,n = X.shape
    
    self.w = np.zeros(n)
    for _ in range(self.nitermax):   
      for i in range(m):
        xi = X[i]
        yi = y[i]

        # Hinge Loss is defined as a piecewise differentiable function
        if yi * np.dot(self.w, xi) < 1:
            self.w = self.w - self.eta * (self.xlambda * self.w - yi * xi)
        else:
            self.w = self.w - self.eta * self.xlambda * self.w
    
    return
  
  def predict(self, Xtest:np.ndarray):
    """ Predict the labels for the given examples based on the weight vector self.w

    Parameters:
    -----------
      Xtest: 2d array of input data

    Returns:
    -------- 
      y: 1d array of predicted labels
    """
    y = np.sign(np.dot(Xtest, self.w))
    return(y)

def load_data() -> (np.ndarray, np.ndarray):
  """ Load and return the breast cancer dataset
  """
  return load_breast_cancer(return_X_y=True)

def main():

  C = int(input("Define penalty coefficient C for margin violations: "))
  # Parameters
  nitermax, eta = 200, 0.1
  
  X, y = load_data()
  print("Data shape:", X.shape, y.shape)

  # Convert the {0,1} output into {-1,+1}
  y = 2 * y -1

  mdata,ndim=X.shape

  # Normalize the input variables
  X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))

  # Fix the random number generator  
  random_state = np.random.seed(12345) 

  # Split the data into 5-folds
  nfold = 5
  cselection_outer = KFold(n_splits=nfold, random_state=random_state, shuffle=False)

  f1_list = []
  # Run cross-validation
  for index_train, index_test in cselection_outer.split(X):
    Xtrain = X[index_train]
    ytrain = y[index_train]
    Xtest = X[index_test]
    ytest = y[index_test]

    # Initialize the SVM classifier
    svm_classifier = svm_primal_cls(C = C, eta = eta, \
      xlambda = 1/C, nitermax = nitermax)
        
    # Run SGD for SVM classifier
    svm_classifier.fit(X=Xtrain, y=ytrain)

    ypred = svm_classifier.predict(Xtest=Xtest)
    
    f1 = f1_score(ytest, ypred)
    f1_list.append(f1)

  # Calculate mean F1 score
  mean_f1 = np.mean(f1_list)
  print(mean_f1)

if __name__ == "__main__":
  main()

