import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''


	# print ("TASK1: INPT", X, "\n\n", Y, "-----")
	transpose_X = np.transpose(X)
	Y = Y.astype(float)
	# X = X.astype(float)
	new_X = np.array([np.ones((np.shape(transpose_X[0])[0],1), dtype=float)])
	new_X = new_X.astype(float)


	# print(np.shape(new_X))
	# std_deviation = []
	for i in range(1, len(transpose_X)):
		p = transpose_X[i]
		if type(p[0]) == type("a"):
			hot_encode = one_hot_encode(p, np.unique(p))
			hot_encode_transpose = np.transpose(hot_encode)

			hot_encode_transpose = hot_encode_transpose.astype(float)
			# print("hot_encode:", hot_encode_transpose[0])

			for z in hot_encode_transpose:
				# print(np.shape(z))
				z = z.reshape((z.shape[0], 1))
				# print(np.shape(z))

				new_X = np.vstack(( new_X, np.array([z])))
		else:	
			temp_mean = np.mean(p)
			temp_std = np.std(p)
			p = p.astype(float)
			p_new = np.array([ (x - temp_mean)/temp_std for x in p])
			p_new = p_new.reshape((p_new.shape[0], 1))
			# print("new_X",p_new)
			# print(np.shape(p_new), np.shape(p), np.shape(new_X))
			# print(p_new, new_X)
			new_X = np.vstack(( new_X, np.array([p_new])))
			# print ("output", np.array([new_X]), "\n", np.array([p_new]), "\n", np.vstack(( new_X, np.array([p_new]))))


	# new_X = np.array(new_X)
	# print(	np.shape(new_X), new_X)

	new_X = np.transpose(new_X)

	# print(	np.shape(new_X), np.shape(Y))
	# print ("TASK1:ouput",new_X[0], Y)
	return new_X[0],Y



	pass

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	# print(np.shape(X), np.shape(2*_lambda*W), np.shape(Y))
	# Y=Y.astype(float)

	# print ("matmul_matrix:", X, W, Y)
	return -2*np.matmul(np.transpose(X), (Y - np.matmul(X, W)))  + 2*_lambda*W
	pass

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	# print (X, "\n\n", Y, "\n-----")
	X = X.astype(float)
	Y = Y.astype(float)
	weights = np.ones([X.shape[1],1], dtype=float)

	# print("ridge" ,X.shape, Y.shape, weights.shape, X[0].shape[0])
	# print(np.shape(X[0]), np.shape(weights))
	# print(X)
	# print(Y)

	for i in range(max_iter):
		grad_ridge_derivative = grad_ridge(weights, X, Y, _lambda)
		if (np.linalg.norm(grad_ridge_derivative) <= epsilon):
			break
		weights = weights - lr*grad_ridge_derivative
	return weights

	pass

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	X = X.astype(float)
	Y = Y.astype(float)


	# X,Y = preprocess(X,Y)


	X_new = np.split(X, k)
	Y_new = np.split(Y,k)

	# print("TASK3: splitX:", X_new, "\nTASK3: splitY:", Y_new)

	sse_array = []
	for z in lambdas:
		temp_sse_array = []
		for p in range(k):
			Xi = X_new[p]
			Yi = Y_new[p]
			temp1 = list(X_new)
			temp2 = list(Y_new)
			# print("tempX, tempY", X_new, Y_new)

			temp1 = np.delete(temp1, p, 0).tolist()
			temp2 = np.delete(temp2, p, 0).tolist()

			tempX = []
			tempY = []

			# print("temp1, temp2", temp1, temp2)
			for i in temp1:
				tempX = tempX + i
			for i in temp2:
				tempY = tempY + i

			# print ("tempX_Y", tempX,"\n\n\n", tempY, "\n\n\n")

			tempX = np.array(tempX, dtype = float)
			tempY = np.array(tempY, dtype=float)
			temp_W = algo(tempX, tempY, z)

			# print ("temp_W", temp_W, temp_W.shape, Xi.shape)
			# z = np.matmul(Xi, temp_W)
			temp_sse = sse(Xi, Yi, temp_W)
			temp_sse_array.append(temp_sse)
		sse_array.append(float(float(sum(temp_sse_array))/float(len(temp_sse_array))))


	return sse_array
	pass

def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''	

	# https://stats.stackexchange.com/questions/347796/coordinate-descent-for-lasso



	X = X.astype(float)
	Y = Y.astype(float)
	weights = np.ones([X.shape[1],1], dtype=float)	

	Z = np.transpose(X)
	Z = np.array([sum(i**2)  for i in Z])
	# print("lasso" ,X.shape, Y.shape, weights.shape, Z.shape)
	# max_iter = 10

	X_transpose = np.transpose(X)
	for p in range(1000):
		# print (p, max_iter)
		# print("SSE actual = ", np.linalg.norm(Y.flatten() - X@weights))
		for q in range(len(X_transpose)):
			weights[q] = 0
			X_new = X_transpose[q]
			intermediate= Y - np.matmul(X,  weights)
			rho = np.matmul(np.transpose(intermediate), X_new)
			# print (rho.shape, X.shape, Y.shape, intermediate.shape, weights.shape, mult_temp.shape)
			# print ("rho", rho)
			if(Z[q] == 0):
				weights[q] = 0
			else:
				if (rho[0] < -_lambda):
					temp_weight = float(rho[0])/Z[q] + float(_lambda)/ float(Z[q])
				elif (rho[0] > _lambda):
					temp_weight = float(rho[0])/Z[q] - float(_lambda)/ float(Z[q])
				else:
					temp_weight = 0	

				# if temp_weight < 0:
				# 	if temp_weight > 0:
				# 		temp_weight = 0

				weights[q] = temp_weight
				# weights[q] = float(rho + _lambda)/ float(Zj)

			# return float(rho - _lambda)/ float(Zj)


				# weights[q] = S(rho[q], _lambda, Z[q][0])
	# print (weights)
	weights = np.reshape(weights, (weights.shape[0], 1))
	return weights

	pass

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	print ("in_task")
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	lambdas = [2, 4, 6, 8, 10, 12, 14, 16] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, ridge_grad_descent)
	plot_kfold(lambdas, scores)