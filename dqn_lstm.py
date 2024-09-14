import os
import pandas as pd
import datetime as datetime
import cryptocompare
import csv
import numpy as np
import random
import math
import time
from requests.exceptions import ConnectionError, Timeout, RequestException
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Attention

cryptocompare.cryptocompare._set_api_key_parameter("KEY_PLACEHOLDER")
alpaca_api_key = 'KEY_PLACEHOLDER'
alpaca_secret_key = 'KEY_PLACEHOLDER'

data = cryptocompare.get_historical_price_hour('ETH', 'USD', limit=999, exchange='CCCAGG', toTs=datetime.datetime.now())

headers_to_exclude = ['conversionType', 'conversionSymbol']
cleaned_data = [{k: v for k, v in item.items() if k not in headers_to_exclude} for item in data]

csv_file = "eth_1_year_data.csv"

with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=cleaned_data[0].keys())
    writer.writeheader()
    writer.writerows(cleaned_data)

df = pd.read_csv(csv_file)
df['50sma'] = df['close'].rolling(50).mean()
df['20sma'] = df['close'].rolling(20).mean()
df.dropna(inplace=True)
df.to_csv(csv_file, index=False)

df = pd.read_csv(csv_file)

features = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close', '20sma', '50sma']
df[features] = df[features].ffill()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[features].values)

def create_sequences(data, time_steps=20):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i, 5])
    return np.array(X), np.array(y)

time_steps = 20
X, y = create_sequences(scaled_data, time_steps)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

input_shape = (X_train.shape[1], X_train.shape[2])

def lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out, hidden_state, cell_state = LSTM(50, activation='relu', return_sequences=True, return_state=True)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    query_value_attention_seq = Attention()([lstm_out, lstm_out])
    query_value_attention_concat = Concatenate(axis=-1)([lstm_out, query_value_attention_seq])
    lstm_out_2 = LSTM(50, activation='relu', return_sequences=False)(query_value_attention_concat)
    lstm_out_2 = Dropout(0.2)(lstm_out_2)
    output = Dense(1, activation='linear')(lstm_out_2)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

attention_lstm_model = lstm_model(input_shape)

history = attention_lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

attention_lstm_model.save('attention_lstm_model.h5')

loss = attention_lstm_model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss}')

last_sequence = X_test[-1].reshape((1, time_steps, X_test.shape[2]))
predicted_price_scaled = attention_lstm_model.predict(last_sequence)

predicted_price = scaler.inverse_transform(np.concatenate((predicted_price_scaled, np.zeros((predicted_price_scaled.shape[0], len(features)-1))), axis=1))[:,0]

current_price_data = cryptocompare.get_price('ETH', currency='USD')
current_price = current_price_data['ETH']['USD']

percentage_error = abs(current_price - predicted_price) / current_price * 100

print(f"Predicted ETH Price: {predicted_price[0]} USD")
print(f"Current ETH Price: {current_price} USD")
print(f"Prediction Error: {percentage_error[0]:.2f}%")

trading_client = TradingClient(alpaca_api_key, alpaca_secret_key, paper=True)

class MarketEnvironment:
    def __init__(self, csv_file_path, initial_balance=100000, lookback_window_size=20):
        self.data = pd.read_csv(csv_file_path)
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.state_size = (lookback_window_size, self.data.shape[1] + 2)
        self.balance = initial_balance
        self.holdings = 0
        self.total_portfolio_value = self.balance
        self.done = False
        self.current_step = 0
        self.alpaca_api = trading_client
        self.last_predicted_change = None  # Initialize last successful predicted change

    def reset(self):
        self.balance = self.initial_balance
        self.holdings = 0
        self.total_portfolio_value = self.balance
        self.current_step = 0
        self.done = False
        return self.get_state(self.current_step)

    def step(self, action):
        print(f"Action taken: {action}")
        assert action in [0, 1, 2]
        self._update_portfolio()
        current_price = self._get_current_price()
        reward = 0
        if action == 1:
            reward = self._buy(current_price)
        elif action == 2 and self.holdings > 0:
            reward = self._sell(current_price)
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True
        self.total_portfolio_value = self.balance + self.holdings * current_price
        next_state = self.get_state(self.current_step)
        print(f"Balance: {self.balance}, Holdings: {self.holdings}, Portfolio Value: {self.total_portfolio_value}")
        return next_state, reward, self.done, {}

    def _update_portfolio(self, retries=5, delay=2):
        attempt = 0
        while attempt < retries:
            try:
                self.balance = float(self.alpaca_api.get_account().cash)
                return
            except (ConnectionError, Timeout, RequestException) as e:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                attempt += 1
                time.sleep(delay)
                delay *= 2  # Exponential backoff
        raise Exception("Failed to update portfolio after multiple attempts")

    def _get_current_price(self, retries=5, delay=2):
        for attempt in range(retries):
            try:
                price_data = cryptocompare.get_price('ETH', currency='USD')
                if price_data and 'ETH' in price_data and 'USD' in price_data['ETH']:
                    return price_data['ETH']['USD']
                else:
                    print(f"Attempt {attempt + 1} failed: No valid price data returned. Retrying in {delay} seconds...")
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        raise Exception(f"Failed to retrieve current price after {retries} attempts")

    def _buy(self, current_price):
        amount_to_invest = math.floor(self.balance * 0.01)
        print(f"Amount to invest: {amount_to_invest} USD")
        if amount_to_invest > 0 and self.holdings == 0:
            try:
                order = MarketOrderRequest(
                    symbol="ETHUSD",
                    notional=amount_to_invest,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                )
                self.alpaca_api.submit_order(order)
                self.holdings = amount_to_invest / current_price
                self.balance -= amount_to_invest
                print(f"Bought at price: {current_price}, Holdings after buy: {self.holdings}, Balance after buy: {self.balance}")
                return 1
            except Exception as e:
                print(f"Failed to submit buy order: {e}")
                return 0
        else:
            print("Not enough balance to buy or already holding.")
            return 0

    def _sell(self, current_price):
        if self.holdings > 0:
            # Calculate the sell amount based on all holdings at current market price
            sell_amount = self.holdings * current_price
            print(f"Calculated sell amount: {sell_amount} USD, Current holdings: {self.holdings} ETH")
            if sell_amount > 0:
                try:
                    # Create and submit a market order to sell all holdings
                    order = MarketOrderRequest(
                        symbol="ETHUSD",
                        qty=self.holdings,  # Selling all holdings
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC
                    )
                    self.alpaca_api.submit_order(order)
                    
                    # Update balance and holdings
                    self.balance += sell_amount
                    self.holdings = 0
                    reward = self.balance - self.initial_balance
                    print(f"Successfully sold all holdings at price: {current_price}, New balance: {self.balance}")
                    return reward
                except Exception as e:
                    print(f"Failed to submit sell order: {e}")
                    return 0
            else:
                print("Sell amount too small.")
                return 0
        else:
            print("No holdings to sell.")
            return 0

    def get_state(self, current_step):
        window_start = max(current_step - self.lookback_window_size, 0)
        window_end = current_step + 1
        window_data = self.data.iloc[window_start:window_end].values
        print(f"window_data shape: {window_data.shape}")
        balance_and_holdings = np.array([[self.balance, self.holdings]] * window_data.shape[0])
        print(f"balance_and_holdings shape: {balance_and_holdings.shape}")
        state = np.concatenate((window_data, balance_and_holdings), axis=1)
        print(f"state shape: {state.shape}")
        return state

class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Memory for experience replay
        self.gamma = 0.95  # Discount rate for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration probability
        self.learning_rate = 0.001  # Learning rate
        self.model = self._build_model()  # The DQN model
        self.lstm_model = load_model('attention_lstm_model.h5')

    def _build_model(self):
        """Builds a NN model."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, current_price, predicted_price, percentage_error, use_lstm=True):
        act_values = self.model.predict(state)
        
        if random.uniform(0, 1) <= self.epsilon:
            # Exploration: Choose a random action
            dqn_action = random.randrange(self.action_size)
            print(f"Exploration: Taking random action {dqn_action}")
        else:
            # Exploitation: Choose the best action based on Q-values
            dqn_action = np.argmax(act_values[0])
            print(f"Exploitation: Taking action {dqn_action} based on Q-values")

        final_action = dqn_action

        if use_lstm:
            try:
                predicted_change = (predicted_price - current_price) / current_price
                predicted_change = predicted_change.item()  # Ensure it's a scalar value
                self.last_predicted_change = predicted_change  # Update last successful predicted change
                print(f"Predicted change: {predicted_change * 100:.2f}%")
            except Exception as e:
                print(f"Error predicting change: {e}. Using last known predicted change.")
                predicted_change = self.last_predicted_change if self.last_predicted_change is not None else 0

            # Ensure percentage_error is a float
            percentage_error = float(percentage_error)
            print(f"Percentage error: {percentage_error:.2f}%")

            # Define the significance thresholds based on percentage_error and corresponding biases
            if percentage_error <= 5.0:
                threshold = 0.05  
                bias = 0.025
            elif percentage_error <= 10.0:
                threshold = 0.10  
                bias = 0.05
            else:
                threshold = 0.15  
                bias = 0.075

            # Scale the bias based on the maximum absolute Q-value
            max_abs_q_value = np.max(np.abs(act_values[0]))
            scaled_bias = bias * max_abs_q_value
            print(f"Using threshold: {threshold * 100:.2f}%, Bias: {bias}, Scaled Bias: {scaled_bias}")

            # Print Q-values before applying bias
            print(f"Q-values before bias: {act_values[0]}")

            # Apply bias based on the predicted change direction
            if predicted_change > 0:
                act_values[0][1] += scaled_bias  # Bias towards buy
                print(f"LSTM predicts price rise, increasing buy action value by {scaled_bias}")
            elif predicted_change < 0:
                act_values[0][2] += scaled_bias  # Bias towards sell
                print(f"LSTM predicts price drop, increasing sell action value by {scaled_bias}")

            # Print Q-values after applying bias
            print(f"Q-values after bias: {act_values[0]}")

            # Choose action based on possibly adjusted Q-values
            final_action = np.argmax(act_values[0])
            print(f"Final action based on adjusted Q-values: {final_action}")

        return final_action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if reward is None:
                print("Warning: Received None as reward")
                continue
            target = reward
            next_state_pred = self.model.predict(next_state)
            if next_state_pred is not None and len(next_state_pred) > 0:
                max_next_state_pred = np.amax(next_state_pred[0])
                target += self.gamma * max_next_state_pred
            else:
                print("Invalid prediction for next_state:", next_state_pred)
                continue
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Loads a saved model."""
        self.model.load_weights(name)

    def save(self, name):
        """Saves the model."""
        self.model.save_weights(name)

# MAIN LOOP

initial_balance = 100000
lookback_window_size = 1

market_environment = MarketEnvironment(csv_file, initial_balance, lookback_window_size)

state_size = market_environment.state_size[0] * market_environment.state_size[1]
action_size = 3

trading_agent = TradingAgent(state_size, action_size)

num_episodes = 8640
timesteps_per_episode = len(data) - lookback_window_size
print(f"timesteps_per_episode: {timesteps_per_episode}")

batch_size = 32

for e in range(num_episodes):
    state = market_environment.reset()
    state = np.reshape(state, (1, -1))
    for time in range(timesteps_per_episode):
        current_price = market_environment._get_current_price()
        print(f"Current Price: {current_price}")
        
        action = trading_agent.act(state, current_price, predicted_price, percentage_error)
        print(f"Action: {action}")
        
        next_state, reward, done, _ = market_environment.step(action)
        print(f"Reward: {reward}, Done: {done}")
        trading_agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e + 1}/{num_episodes}, Total Portfolio Value: {market_environment.total_portfolio_value}, P/L: {market_environment.total_portfolio_value - initial_balance}")
            break
        if len(trading_agent.memory) > batch_size:
            print(f"Training on memory batch of size: {batch_size}")
            trading_agent.replay(batch_size)
    if e % 10 == 0:
        trading_agent.save(f"trading_agent_{e}.weights.h5")

"""
MODEL TIMING

The total time per episode in seconds would be 0.005 seconds * 999 timesteps = 4.995 seconds.

For 100 episodes, the total time would be 4.995 seconds/episode * 100 episodes = 499.5 seconds.

43200 seconds = 24 hours. 43200/499.5 = ~86.4. 86.4 x 100 = 8640 episodes to run ~24 hours.


HYPERPARAMETERS

Time Steps (time_steps): Used in the sequence creation function, it defines the number of historical data points the LSTM model will consider for predicting the next value. It's set to 20, meaning the model looks at the last 20 data points to make a prediction.

Data Split (split): Determines the ratio of training to testing data. It's set to 80% for training and the remaining 20% for testing.

LSTM Units: Inside the lstm_model function, LSTM layers with 50 units are defined, which signifies the dimensionality of the output space.

Dropout Rate: Dropout layers with a rate of 0.2 are used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.

Dense Layer Units: The dense layers in the LSTM model have 24 units each, which is another hyperparameter defining the output dimension of the layer.

Batch Size for Training LSTM Model (batch_size): The LSTM model is trained with a batch size of 32, which means 32 sequences are passed through the network at once.

Epochs for Training LSTM Model: The model is trained for 100 epochs, where one epoch means that the entire dataset is passed forward and backward through the neural network once.

Lookback Window Size (lookback_window_size): Set to 20, this defines how many past observations the trading environment will return as the state.

State Size (state_size): Calculated based on the lookback_window_size and the number of features in the data. It represents the shape of the input that the trading agent will receive.

Gamma (gamma): This is the discount factor for future rewards in the DQN agent, set to 0.95.

Epsilon (epsilon): The exploration rate for the agent, starting at 1.0 (100% exploration).

Epsilon Minimum (epsilon_min): The minimum exploration rate, set to 0.01.

Epsilon Decay (epsilon_decay): The rate at which the exploration rate decays, set to 0.995.

Learning Rate (learning_rate): For the Adam optimizer in the DQN, it's set to 0.001.

Number of Episodes (num_episodes): The number of episodes for training the DQN model. It's initially set to 1800, which indicates how many times the training process will iterate over the entire dataset.

Timesteps Per Episode (timesteps_per_episode): The number of steps the agent will take in the environment per episode. It's initially logged as 999, which is derived from the available historical price hours minus the lookback_window_size.
"""
