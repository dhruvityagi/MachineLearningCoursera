function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



n = length(theta);
temp = 0;
for i=1:m
	temp = temp + (-y(i)*log(sigmoid(X(i,:) * theta)) - (1-y(i))*log(1-sigmoid(X(i,:) * theta)));
end

regParam = 0;
for j=2:n
	regParam = regParam + theta(j) * theta(j);
end

J = temp/m + (regParam * lambda)/(2*m);

%Calculating grad(1) which is for 0
temp1 = 0;
for i=1:m
	temp1 = temp1 + (sigmoid(X(i,:)*theta()) - y(i)) * X(i,1);
end
grad(1) = temp1 / m;

%Calculating grad(j) for j>=1
for j=2:n
	temp1 = 0;
	for i=1:m
		temp1 = temp1 + (sigmoid(X(i,:)*theta()) - y(i)) * X(i,j);
	end
	grad(j) = (temp1 / m) + (lambda * theta(j))/m;
end



% =============================================================

end
