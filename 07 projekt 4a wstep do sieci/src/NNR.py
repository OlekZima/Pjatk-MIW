import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from typing import List
from typing import Callable


class NeuralNetworkRegression:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation_function: Callable,
        activation_function_derivative: Callable,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

        self.weights_input_hidden: np.ndarray = np.random.randn(
            self.input_size, self.hidden_size
        )

        self.weights_hidden_output: np.ndarray = np.random.randn(
            self.hidden_size, self.output_size
        )

        self.bias_hidden: np.ndarray = np.zeros(self.hidden_size, dtype=np.float32)
        self.bias_output: np.ndarray = np.zeros(self.output_size, dtype=np.float32)

        self.history_mse: List[float] = []
        self.history_r2: List[float] = []

    @staticmethod
    def sigmoid(x_data: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x_data))

    @staticmethod
    def sigmoid_derivative(x_data: np.ndarray) -> np.ndarray:
        return NeuralNetworkRegression.sigmoid(x_data) * (
            1 - NeuralNetworkRegression.sigmoid(x_data)
        )

    @staticmethod
    def relu(x_data: np.ndarray) -> np.ndarray:
        return x_data * (x_data >= 0)

    @staticmethod
    def relu_derivative(x_data: np.ndarray) -> np.ndarray:
        return (x_data >= 0).astype(int)

    @staticmethod
    def tanh(x_data: np.ndarray) -> np.ndarray:
        return np.tanh(x_data)

    @staticmethod
    def tanh_derivative(x_data: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x_data) ** 2

    @staticmethod
    def linear(x_data: np.ndarray) -> np.ndarray:
        return x_data

    @staticmethod
    def linear_derivative(x_data: np.ndarray) -> np.ndarray:
        return np.ones_like(x_data)

    @staticmethod
    def softplus(x_data: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x_data))

    @staticmethod
    def softplus_derivative(x_data: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x_data))

    def _pass_forward(self, x_data: np.ndarray) -> tuple:
        hidden_input = x_data @ self.weights_input_hidden + self.bias_hidden
        hidden_output: np.ndarray = self.activation_function(hidden_input)
        output = hidden_output @ self.weights_hidden_output + self.bias_output
        return (
            hidden_input,
            hidden_output,
            output,
        )

    def _pass_backward(
        self, x_data, y_data, forward_result, learning_rate: float, reg_l2
    ) -> None:
        hidden_input, hidden_output, output = forward_result

        error = output - y_data
        d_output = error / x_data.shape[0]

        d_weights_hidden_output = (
            hidden_output.T @ d_output + reg_l2 * self.weights_hidden_output
        )
        d_bias_output = np.sum(d_output, axis=0)

        d_hidden_output = d_output @ self.weights_hidden_output.T
        d_hidden_input = d_hidden_output * self.activation_function_derivative(
            hidden_input
        )

        d_weights_input_hidden = (
            x_data.T @ d_hidden_input + reg_l2 * self.weights_input_hidden
        )
        d_bias_hidden = np.sum(d_hidden_input, axis=0)

        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output
        self.bias_hidden -= learning_rate * d_bias_hidden
        self.bias_output -= learning_rate * d_bias_output

    def train(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        epochs: int,
        learning_rate: float,
        reg_l2,
    ) -> int:
        self.epoch = epochs
        self.learning_rate = learning_rate
        self.reg_l2 = reg_l2
        self.epoch_num_c = 0

        for epoch_num in range(1, epochs + 1):
            self.epoch_num_c += 1

            output = self._pass_forward(x_data)
            self._pass_backward(x_data, y_data, output, learning_rate, reg_l2)

            output = output[-1]
            mse = mean_squared_error(y_data, output)
            r2 = r2_score(y_data, output)
            self.history_mse.append(mse)
            self.history_r2.append(r2)
            if epoch_num % (epochs // 10) == 0:
                print(f"Epoch {epoch_num} of {epochs}")
                print(f"MSE: {mse}, R2: {r2}")
                print("=" * 80)
            if r2 >= 0.95:
                break
        print("Fitting complete!")

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        return self._pass_forward(x_data)[-1]
