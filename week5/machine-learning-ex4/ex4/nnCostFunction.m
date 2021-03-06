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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

%Unregularized Cost function
% Add ones to the X
X = [ones(m, 1) X];

% Calculate hidden layer activation units
% Layer 2
hiddenLayerActivationUnit = zeros(m, hidden_layer_size);
hiddenLayerZ = X * Theta1';
hiddenLayerActivationUnit = sigmoid(hiddenLayerZ);


% Layer 3 - Hypothesis
hiddenLayerActivationUnit = [ones(m, 1) hiddenLayerActivationUnit];
hypothesis = zeros(m,num_labels);
hypothesis = sigmoid(hiddenLayerActivationUnit * Theta2');

%Vector Y
vecY = zeros(m,num_labels);
for i=1:m
	indexForOne = y(i);
	vecY(i,indexForOne) = 1; 
end
	
%Cost function
temp = 0;
for i = 1:m
	for k = 1:num_labels
		temp1 = - (vecY(i,k) * log(hypothesis(i,k)));
		temp2 = - ((1 - vecY(i,k)) * (log(1 - hypothesis(i,k))));
		temp = temp + temp1 + temp2;
	end
end

J = temp/m;

%Regularized Cost function

regularizedTermForTheta1 = 0;
regularizedTermForTheta2 = 0;

for j = 1:hidden_layer_size
	regularizedTermForTheta1 = regularizedTermForTheta1 + sum(Theta1(j,2:end) .* Theta1(j,2:end));
end

for j = 1:num_labels
	regularizedTermForTheta2 = regularizedTermForTheta2 + sum(Theta2(j,2:end) .* Theta2(j,2:end));
end

J = J + ((regularizedTermForTheta1 + regularizedTermForTheta2) * (lambda / (2 * m)));

% Back Propagation
% Step 1 - Already done above for forward pass

% Step 2 - Delta in output unit
delk3 = hypothesis .- vecY;
delK3Size = size(delk3);

a = size(X);
b = size(Theta1);
c = size(Theta2);
d = size(hiddenLayerZ);
e = size(vecY);

% Step 3 - Delta in hidden layer
delk2 = (delk3 * Theta2(:,2:end)) .* sigmoidGradient(hiddenLayerZ);
delk2size = size(delk2);

% Step 4 - Accumulate gradient
Delta2 = delk3' * hiddenLayerActivationUnit;

Delta1 = delk2' * X;

Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;


% Regularized gradient
tempRegTheta1 = zeros(size(Theta1));
tempRegTheta2 = zeros(size(Theta2));
tempRegTheta1(:,2:end) = ((lambda .* Theta1(:,2:end)) ./ m);
tempRegTheta2(:,2:end) = ((lambda .* Theta2(:,2:end)) ./ m);
Theta1_grad = Theta1_grad + tempRegTheta1;
Theta2_grad = Theta2_grad + tempRegTheta2;





		
	


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
