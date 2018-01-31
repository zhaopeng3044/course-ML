function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

row_and_col = size(z);

for i=1:row_and_col(1)
	for j=1:row_and_col(2)
		g(i,j) = 1/(1+e^-z(i,j));
	end
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================


end
