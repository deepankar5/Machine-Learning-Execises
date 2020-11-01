function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X = [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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
% a(0) = X (which is the input node)
% z(2) which is the first node
% z(2)= thetha1 * a(0) or thetha1 * X
z_2 = X * Theta1';
% a_2 = g(z(2)) 
a_2 = sigmoid(z_2);
% add one extra bais value a_2_0 = 1
a_2_m = size(a_2);
a_2 = [ones(a_2_m,1) a_2];
%z(3) = will take input node as a_2
z_3 = a_2 * Theta2';
% a_3 = g(z_3), which is the logistical regration
a_3 = sigmoid(z_3);

% this will give an array of size(X,1) X num_labels, and p will
% tell the position of the maximum value

[x, p] = max(a_3, [] , 2);


% =========================================================================


end
