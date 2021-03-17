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