function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

%compute forward result
tmp_ones = ones(m, 1);
X = [tmp_ones X];
z1 = X * Theta1';
a1 = [tmp_ones sigmoid(z1)];
z2 = a1 * Theta2';
a2 = sigmoid(z2);

% You need to return the following variables correctly 
J = 0;
for i=1:m
    for j=1:num_labels
        tmp_y = y(i) == j;
        tmp_output = a2(i, j);
        J += (-tmp_y * log(tmp_output) - (1 - tmp_y) * log(1 - tmp_output));
    end
end

J = J / m;

reg_sum = sum(sum(Theta1(:, 2:end) .^ 2));
reg_sum += sum(sum(Theta2(:, 2:end) .^ 2));

J += lambda * reg_sum / (2 * m);

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

sum_theta1 = zeros(hidden_layer_size, (input_layer_size+1));
sum_theta2 = zeros(num_labels, (hidden_layer_size+1));

for i=1:m
    y_i = zeros(num_labels, 1); % 10 X 1
    y_i(y(i)) = 1;
    d_3 = a2(i, :)' - y_i; % 10 X 1
    d_2 = Theta2' * d_3 .*  sigmoidGradient([1 z1(i, :)]'); % 26 X 1
    d_2 = d_2(2:end);
    sum_theta2 += (d_3 * a1(i, :)); % 10 X 26
    sum_theta1 += (d_2 * X(i, :)); % 25 * 401
end

sum_theta2 /= m;
sum_theta1 /= m;
grad = [sum_theta1(:); sum_theta2(:)];
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
