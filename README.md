# CTB

![Figure_1](https://github.com/bitsbard/ctb/assets/114309008/2c1182a1-af2f-41ee-b76e-c8490dc97132)

This model implements a deep reinforcement learning (RL) agent to trade Ethereum (ETH) using historical price data. It combines a Long Short-Term Memory (LSTM) model with attention mechanisms for price prediction and a deep Q-learning network for trading decisions. Here's a comprehensive breakdown of the code and its components:

### Imports and API Keys Setup

- The notebook starts by importing various libraries essential for data manipulation (`pandas`, `numpy`), machine learning (`tensorflow.keras`), and trading (`cryptocompare`, `alpaca`).
- API keys for `cryptocompare` and `Alpaca` are set up to fetch historical price data and enable trading functionalities.

### Fetch Historical Data

- The notebook fetches historical hourly price data for ETH from `cryptocompare` and cleans unwanted values (e.g., `conversionType`, `conversionSymbol`).
- This cleaned data is then saved to a CSV file, `eth_1_year_data.csv`.

### Calculate Simple Moving Averages (SMA)

- The notebook calculates 50-period and 20-period SMAs based on the historical closing prices and adds these SMAs to the CSV.
- Rows with NaN values resulting from SMA calculations are dropped to ensure clean data for model training.

### Data Preprocessing

- Data is loaded from the CSV file, and missing values are forward-filled using `fillna`.
- The data is then normalized using `MinMaxScaler` to scale the features between 0 and 1.
- A sequence creation function is defined to generate input sequences (`X`) and target variables (`y`) for model training. Each sequence consists of 20 time steps of historical data, and the target variable is the closing price at the next time step.

### Train LSTM Model with Attention Mechanism

- An LSTM model with attention layers is defined to predict future prices. Attention mechanisms help the model focus on relevant parts of the input sequence.
- The model is trained using the prepared sequences, evaluated for its performance, and saved as `attention_lstm_model.h5`.

### Make Price Prediction

- After training, the LSTM model is used to predict the next closing price of ETH using the latest test sequence.
- The predicted price is compared with the current price fetched from `cryptocompare`, and the prediction error is calculated.

### Market Environment and Trading Agent Classes

- **MarketEnvironment**: Manages the trading environment, including balance, holdings, and portfolio value. It simulates trading actions (buy/sell) based on the current price and updates the state of the environment.
- **TradingAgent**: Implements a deep Q-learning agent that decides whether to buy, sell, or hold ETH based on the state of the market. It uses a neural network for Q-value estimation and an experience replay memory for training.

### Main Training Loop

- The main training loop initializes the market environment and trading agent.
- It runs multiple episodes where the agent interacts with the market environment to learn a trading policy. In each episode, the agent takes actions, receives rewards, and updates its policy using experience replay.
- Periodically, the trading agent's model is saved to ensure progress is not lost.

### Output

- The notebook prints the predicted ETH price, current ETH price, and prediction error.
- Debug statements print shapes of arrays used in state representation to ensure correctness.

### Detailed Explanation of Debug Messages

During execution, the notebook prints debug messages to verify the shapes of arrays used to represent the state of the trading environment. Here’s an explanation of what these shapes mean:

1. **`window_data shape: (2, 9)`**:
   - `window_data` contains historical market data over a certain lookback window.
   - The shape `(2, 9)` indicates that `window_data` consists of 2 rows (time steps) and 9 columns (features).
   - The features might include prices (open, high, low, close), volume, and calculated moving averages (e.g., 20sma, 50sma).

2. **`balance_and_holdings shape: (2, 2)`**:
   - `balance_and_holdings` contains the trader’s current balance and the number of holdings.
   - The shape `(2, 2)` indicates that `balance_and_holdings` consists of 2 rows and 2 columns. Each row corresponds to a time step in the lookback window, and the columns represent the balance and holdings.
   - This repetition ensures that the balance and holdings are considered alongside each time step's market data in the state representation.

3. **`state shape: (2, 11)`**:
   - `state` is the final array representing the state of the trading environment, used as input to the RL model.
   - The shape `(2, 11)` indicates that the `state` consists of 2 rows and 11 columns.
   - The 11 columns are a concatenation of the 9 features from `window_data` and the 2 features from `balance_and_holdings`, providing a comprehensive representation of the market and the trader’s current status.

### Purpose of the Debug Statements

- These debug statements help verify that the state representation is correctly formatted before being fed into the RL model.
- They ensure that the dimensions of the arrays match the expected input shape for the neural network, avoiding potential errors during training or decision-making.

### Summary

The model implements a automated trading system using a combination of LSTM and RL models. The LSTM model provides price predictions, which are incorporated into the state representation used by the RL model. This integrated approach helps the RL model make informed trading decisions, potentially improving trading performance by leveraging both historical price trends and future price predictions. The debug statements are crucial for ensuring the data is correctly prepared and formatted, facilitating the smooth functioning of the trading agent. dqn_lstm.py can be run locally.
