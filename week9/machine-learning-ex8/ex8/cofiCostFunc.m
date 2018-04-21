function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

tempSum = 0;
for i = 1:num_movies
	for j = 1:num_users
		if(R(i,j) == 1)
			thetaXMultiplier = 0;
			for k = 1:num_features
				thetaXMultiplier = thetaXMultiplier + Theta(j,k) * X(i,k);
			end
			tempSum = tempSum + (thetaXMultiplier - Y(i,j)) ^ 2;
		endif 
	end
end
J = tempSum / 2;

% Gradients
for i = 1:num_movies
	for j = 1:num_users
		if(R(i,j) == 1)
			thetaXMultiplier = 0;
			for k = 1: num_features
				thetaXMultiplier = thetaXMultiplier + Theta(j,k) * X(i,k);
			end
			X_grad(i,:) = X_grad(i,:) + (thetaXMultiplier - Y(i,j)) .* Theta(j,:);
		endif
	end
end

for j = 1:num_users
	for i = 1:num_movies
		if(R(i,j) == 1)
			thetaXMultiplier = 0;
			for k = 1: num_features
				thetaXMultiplier = thetaXMultiplier + Theta(j,k) * X(i,k);
			end
			Theta_grad(j,:) = Theta_grad(j,:) + (thetaXMultiplier - Y(i,j)) .* X(i,:);
		endif
	end
end

% Regularization
% Cost function

firstSum = 0;
secondSum = 0;
for j = 1:num_users
	firstSum = firstSum + sum(Theta(j,:) .* Theta(j,:));
end
for i = 1:num_movies
	secondSum = secondSum + sum(X(i,:) .* X(i,:));
end
regularizedParam1 = lambda * (firstSum) / 2;
regularizedParam2 = lambda * (secondSum) / 2;
J = J + regularizedParam1 + regularizedParam2;

% Gradient

X_grad = X_grad + lambda .* X;
Theta_grad = Theta_grad + lambda .* Theta;









% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
