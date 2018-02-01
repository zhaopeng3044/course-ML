function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X = [ones(m, 1) X];

lay1_result = 1 ./ ( 1 + e .^ -(X * Theta1'));
lay1_result = [ones(m, 1) lay1_result ];
final_result = 1 ./ (1 + e .^ -(lay1_result * Theta2'));
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

for i=1:m
	max_value = 0;
	index = 1;
	for j=1:num_labels
		if final_result(i, j) > max_value
			max_value = final_result(i, j);
			index = j;
		endif
	end

	p(i) = index;
end
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
