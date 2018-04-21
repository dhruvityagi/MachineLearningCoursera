function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypothesis = X * theta;

J = sum((hypothesis - y) .* (hypothesis - y)) / (2 * m);

% Regularized Cost function
J = J + (lambda * sum(theta(2:end) .* theta(2:end)) / (2 * m));

% Calculating gradient

grad(1) = sum((hypothesis - y) .* X(:,1)) / m;
for j = 2:size(theta)
	grad(j) = (sum((hypothesis - y) .* X(:,j)) / m) + ((lambda .* theta(j)) / m);
end










% =========================================================================

grad = grad(:);

end
