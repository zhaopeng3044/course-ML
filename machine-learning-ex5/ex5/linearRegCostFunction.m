function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
tmp_ones = ones(m, 1);
h_out = X * theta;

% You need to return the following variables correctly 
J = sum((h_out - y) .^ 2) / (2*m) + lambda * sum( theta(2:end, :) .^ 2) / (2 *m);
grad = zeros(size(theta));

grad(1) = sum(h_out -y) / m;

for j=2:size(theta)
    grad(j) = sum((h_out - y ).* X(:, j)) / m + lambda * theta(j) / m;
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
