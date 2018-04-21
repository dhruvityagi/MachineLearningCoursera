function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

CSet = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmaSet = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
maxPredictionValue = 1000000;
for i = 1 : size(CSet)
	for j = 1 : size(sigmaSet)
		Ctemp = CSet(i);
		sigmaTemp = sigmaSet(j);
		model = svmTrain(X, y, Ctemp, @(x1, x2) gaussianKernel(x1, x2, sigmaTemp));
		predictions = svmPredict(model, Xval);
		predictionValue = mean(double(predictions ~= yval));
		if(predictionValue < maxPredictionValue) 
			maxPredictionValue = predictionValue;
			C = Ctemp;
			sigma = sigmaTemp;
		endif
	end
end

a = C
b = sigma



% =========================================================================

end
