# Logistic regression

## Goal

Achieve the goal to detect cat pictures

## Usage

all tools are put in `utils.py`

only need to run the `main.py`

function `optimize` can train the model and `predict` can make a prediction according to input

## Pits

- have to reshape (50,) to (50,1),otherwise this array cannot be transposed

- At first I initialize weights like **W=np.random.random((12288,1))**,but then I found that **Z** is as large as several thousand so that all numbers in A infinitely close 1 after sigmoid due to there has 12288 features in an image. So **W=np.zeros(12288,1)** is better.

- 

  ```python
  #It works if acc.shape == (1,50)
  for i in range(acc.shape[1]):
  	if acc[0,i] == 0:
  		.....
  #It works if acc.shape == (50,)
  for i in acc:
      if acc[i] == 0:
          .....
  ```

- ```python
  params = np.load('myparams.npy',allow_pickle=True).item()
  #item()is required:it create a copy of the specified element of the array as a suitable Python scalar
  #allow_pickle is True:Allows an array of pickled objects stored in an NPY file to be loaded
  ```

  

