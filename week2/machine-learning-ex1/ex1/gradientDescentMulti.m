function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    n = size(X,2);
    X_norm1 = featureNormalize(X(:,2:n));
    X_norm = [ones(m,1),X_norm1()];
    delta = zeros(n,1);
    for k = 1 : n
        for i = 1 : m
            delta(k) = delta(k) + ((X_norm(i, :) * theta) - y(i)) * X_norm(i,k);
        end
    end
    theta = theta .- alpha .* (delta ./ m);
    %disp("Theta:"), disp(theta);
    %disp("Cost function:"), disp(computeCostMulti(X_norm, y, theta));
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X_norm, y, theta);










    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
