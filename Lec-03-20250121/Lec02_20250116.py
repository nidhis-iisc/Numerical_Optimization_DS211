import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#use pandas to load real_estate_dataset
df =pd.read_csv('real_estate_dataset.csv')

#get the number of samples and features
n_samples, n_features = df.shape

#print the number of samples and features
print('The number of samples, features:', n_samples, n_features)


# get the names of the columns
columns = df.columns

# save the column names to file
np.savetxt('real_estate_dataset_columns.txt', columns, fmt='%s')

#use Square_Feet, Garage_Size, Location_score, Distance_to_Center as features
X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']].values

#use Price as the target
y = df['Price'].values

#print X.shape and X.dtype
print(f"Shape of X:, {X.shape}\n")
print('X.dtype:', X.dtype)

# get the number of samples and features in X
n_samples, n_features = X.shape

#Build a linear model to predict the price from the four features in X
# make an array of coefs of the size of n_features+1, initialize to 1
coefs = np.ones(n_features+1)

#Bias= expected value of target variable. When is the prediction exactly equal to bias. 

#let predict the price of each sample in X
predictions_bydefn = X @ coefs[1:] + coefs[0]

#append a column of ones to X
X = np.hstack([np.ones((n_samples, 1)), X])

#predict the price of each sample in X
predictions = X @ coefs


#see if all entries in predictions_bydefn and predictions are the same
is_same = np.allclose(predictions_bydefn, predictions)

print('Are the predictions the same?', is_same)

# calculate the error using predictions and y
error = y - predictions

#calculate the relative error
relative_error = error/y

#calculate the mean of square of errors using a loop
loss_loop = 0
for i in range(n_samples):
    loss_loop += error[i]**2

loss_loop /= n_samples

#calculate the mean of square of errors using matrix operations
loss_matrix = np.transpose(error) @ error / n_samples

#compare the two methods of calculating the mean of square of errors
is_diff = np.allclose(loss_loop, loss_matrix)
print('Are the two methods of calculating the mean of square of errors the same?', is_diff)


#print the size of errors and its L2 norm
print('Size of errors:', error.shape)
print('L2 norm of errors:', np.linalg.norm(error))
print('L2 norm of relative errors:', np.linalg.norm(relative_error))

# Objective function: f(coefs) = 1/n_samples * \sum_{i=1}^{n_samples} (y_i - (coefs[0] + X_i @ coefs[1:]))^2

#What is a solution?
#A solution is a value of coefs that minimizes the objective function.

#How do i find a solution?
# By searching for the coefficients at which the gradient of the objective function is zero.
# or i can set the gradient of the objective function to zero and solve for the coefficients.

#write the loss matrix in terms of the data and coefs

loss = 1/n_samples * (y - X @ coefs).T @ (y - X @ coefs)

#calculate the gradient of the loss with respect to coefs
gradient = -2/n_samples * X.T @ (y - X @ coefs)

#set gradient to zero and solve for coefs
#  X.T @ y = X.T @ X @ coefs
# X.T @ X @ coefs = X.T @ y
# coefs = (X.T @ X)^-1 @ X.T @ y
coefs = np.linalg.inv(X.T @ X) @ X.T @ y


#save the coefs to a csv file
np.savetxt('coefs.csv', coefs, delimiter=',')

#calculate the predictions using the optimal coefs
predictions_model = X @ coefs

#calculate the error using the optimal coefs
error_model = y - predictions_model

#print the L2 norm of the error_model
print('L2 norm of error_model:', np.linalg.norm(error_model))

#calculate and print the relative error using the optimal coefs
relative_error_model = error_model/y
print('L2 norm of relative_error_model:', np.linalg.norm(relative_error_model))

# use all the features in the dataset to build a linear model to predict the price
X_all = df.drop('Price', axis=1).values
y_all = df['Price'].values

# get the number of samples and features in X_all
n_samples_all, n_features_all = X_all.shape
print('The number of samples, features in X_all:', n_samples_all, n_features_all)

#solve the linear model using the normal equation
X_all = np.hstack([np.ones((n_samples_all, 1)), X_all])
coefs_all = np.linalg.inv(X_all.T @ X_all) @ X_all.T @ y

#save coefficients to a csv file
np.savetxt('coefs_all.csv', coefs_all, delimiter=',')


#calculate the rank of X_all.T @ X_all
rank = np.linalg.matrix_rank(X_all.T @ X_all)
print('Rank of X_all.T @ X_all:', rank)


#solve the normal equation using matrix decomposition
Q, R = np.linalg.qr(X_all)

#print shape of Q and R
print('Shape of Q:', Q.shape)
print('Shape of R:', R.shape)

#write R to file names R.csv
np.savetxt('R.csv', R, delimiter=',')


#R.coeffs = b

sol = Q.T @ Q # Identity matrix
np.savetxt('sol.csv', sol, delimiter=',')

# X_all = QR
#X_all.T @ X_all = R.T @ Q.T @ Q @ R = R.T @ R
# X_all @ y = R.T @ Q.T @ y
# R @ coefs_all = Q.T @ y

b = Q.T @ y

print('Shape of b:', b.shape)
print('Shape of R:', R.shape)

#coeff_qr = np.linalg.inv(R) @ b
# loop to solve R*coeffs = b using back substitution
coeffs_qr = np.zeros(n_features_all+1)
for i in range(n_features_all, -1, -1):
    coeffs_qr[i] = b[i]
    for j in range(i+1, n_features_all+1):
        coeffs_qr[i] -= R[i, j] * coeffs_qr[j]
    coeffs_qr[i] /= R[i, i]

#save the coefficients to a csv file
np.savetxt('coeffs_qr.csv', coeffs_qr, delimiter=',')

#perform prediction using the optimal coefficients
predictions_qr = X_all @ coeffs_qr

#calculate the error using the optimal coefficients
error_qr = y - predictions_qr

#calculate the L2 norm of the error
l2_norm_error_qr = np.linalg.norm(error_qr)

#calculate the relative error using the optimal coefficients
relative_error_qr = error_qr/y

#calculate the L2 norm of the relative error
l2_norm_relative_error_qr = np.linalg.norm(relative_error_qr)

#print the L2 norm of the error and the L2 norm of the relative error
print('L2 norm of error_qr:', l2_norm_error_qr)
print('L2 norm of relative_error_qr:', l2_norm_relative_error_qr)


#solve the normal equation using the SVD
U, S, Vt = np.linalg.svd(X_all, full_matrices=False)

#eigen decompostion off a sqauare matrix 
# A = V @ D @ V.T
# A^-1 = V @ D^-1 @ V.T
# X_all*coeffs = y
# A = X_all^T @ X_all

# normal equation: X_all.T @ X_all @ coeffs = X_all.T @ y
# Xdagger = (X_all.T @ X_all)^-1 @ X_all.T


#find the inverse of X in the least square sense
# psuedo inverse of X = V @ D^-1 @ U.T

#to complete calculate the coeffs_svd using the psuedo inverse of X
coeffs_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y

#save the coefficients to a csv file
np.savetxt('coeffs_svd.csv', coeffs_svd, delimiter=',')

#perform prediction using the optimal coefficients
predictions_svd = X_all @ coeffs_svd

#calculate the error using the optimal coefficients
error_svd = y - predictions_svd

#calculate the L2 norm of the error
l2_norm_error_svd = np.linalg.norm(error_svd)

#calculate the relative error using the optimal coefficients
relative_error_svd = error_svd/y

#calculate the L2 norm of the relative error
l2_norm_relative_error_svd = np.linalg.norm(relative_error_svd)

#print the L2 norm of the error and the L2 norm of the relative error
print('L2 norm of error_svd:', l2_norm_error_svd)
print('L2 norm of relative_error_svd:', l2_norm_relative_error_svd)


U, S, Vt = np.linalg.svd(X_all, full_matrices=False)

#solve for X_all_svd @ coeffs_svd = y
#normal equation : X_all_svd.T @ X_all_svd @ coeffs_svd = X_all_svd.T @ y
# replace X_all_svd with U @ np.diag(S) @ Vt
#Vt^T @ np.diag(S)^2 @ Vt @ coeffs_svd = Vt^T @ np.diag(S) @ U.T @ y
# np.diag(S)^2 @ Vt @ coeffs_svd = np.diag(S) @ U.T @ y
# coeffs = Vt @ np.diag(S)^-1 @ U.T @ y

coeffs_svd = Vt.T @ np.diag(1/S)@ U.T @ y
coeff_svd_pinv = np.linalg.pinv(X_all) @ y

#save the coefficients to a csv file
np.savetxt('coeffs_svd.csv', coeffs_svd, delimiter=',')
np.savetxt('coeffs_svd_pinv.csv', coeff_svd_pinv, delimiter=',')


#X_1 = X_all[: ,0:1]
#coeffs_1 = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ y
# if it were rank def as inv(X_all.T @ X_all) would not exist, QR would not work, only SVD would work
#pinv is pseudo inverse (rcond = 1e-15) is the smallest singular value that is considered non zero
#least squares for tall matrices, SVD is the best


#plot the data on X_all[:, 1] vs y
#also plot the regression line with only X_all[:, 0] and X_all[:, 1] as the features
#first make X[:,1] as np.arrange between min and max of X[:,1]
#then calculate the predictions using the optimal coefficients
#plot the data and the regression line
# X_feature = np.arange(np.min(X_1[:, 1]), np.max(X_1[:, 1]), 0.01)
# plt.scatter(X_all[:, 1], y, label='Data')
# plt.plot(X_feature,  X_feature * coeffs_1[1], label='Regression Line', c='Red')
# plt.xlabel('Square Feet')
# plt.ylabel('Price')
# plt.title('Price vs Square Feet')
# plt.legend()
# plt.show()
# plt.savefig('Price_vs_Square_Feet.png')


#use X as only square feet to build a linear mnodel to predict price
X = df[['Square_Feet']].values
y = df['Price'].values
X= np.hstack([np.ones((n_samples, 1)), X])

coeff_1 = np.linalg.inv(X.T @ X) @ X.T @ y
predictions_1 = X @ coeff_1
X_feature = np.arange(np.min(X[:,1]), np.max(X[:,1]), 10)
print("min of X[:,1]", np.min(X[:,1]))
print("max of X[:,1]", np.max(X[:,1]))
# pad the X_feature with ones
X_feature = np.hstack([np.ones((X_feature.shape[0], 1)), X_feature.reshape(-1, 1)])
plt.scatter(X[:,1], y, label='Data')
plt.plot(X_feature[:,1], X_feature  @ coeff_1, label='Regression Line', c='Red')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Price vs Square Feet')
plt.legend()
plt.show()
plt.savefig('Price_vs_Square_Feet.png')





#least squares can be formulated in two ways:
#1. minimize the sum of the squares of the errors along y axis
#2. minimize the sum of the squares of the errors along the normal to the line - orthogonal regression
#4th section of book















