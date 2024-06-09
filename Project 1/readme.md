Project 1
data:yfinance
UI pyqt
Training Pytorch


Outline:

- UI
    - Main window
    - Search for stock and pass ticker to backend
    - displa downlaoding of ticker info
    - Have preloaded sqlite db stock data to choose from
    - have start training button to call ml model to train
        - Loss per epoch
        - Neural network visual updated every 5 steps
        - get weights in layer fashion and visualize one layer at a time for pass through effect
    - pause and resume button for training
    - change NN architecture by adding/removing layers adjusting heights per hidden units, adjust and change lr, activitation functions,etc
 
-Backend
- download stock ticker from UI input
- pass stock ticker info to ml model to train on
- pass weights/Loss to NN visual and loss plot
