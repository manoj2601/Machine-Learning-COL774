import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt 
from numpy import genfromtxt
# import plotly.graph_objects as go

def sigmoid_function(theta, X):
	h_theta = np.matmul(X, theta)
	h_theta = np.multiply(h_theta, (-1))
	h_theta = np.exp(h_theta)
	h_theta[:] += 1
	h_theta[:] = 1/h_theta[:]
	return h_theta


def log_likelihood(X, y, theta):
	m = len(y)
	H = sigmoid_function(theta, X)
	H_2 = np.subtract(np.ones(len(H)), H)
	H_2 = np.log(H_2)
	H_1 = np.log(H)
	L_1 = np.matmul(np.transpose(y), H_1)
	L_2 = np.matmul(np.transpose(np.subtract(np.ones(len(y)), y)), H_2)
	return (L_1 + L_2)

def isConversed(theta0, theta1):
	X = np.abs(np.subtract(theta0, theta1))
	y = np.sum(X)
	if(y < 0.001):
		return True
	else:
		return False

def findx2(theta, X):
	first = np.ones(len(X))
	first = np.multiply(first, (-1*theta[0]/theta[2]))
	second = np.multiply(X, (-1*theta[1]/theta[2]))
	return np.add(first, second)


def HessianMatrix(X, theta):
	sig = sigmoid_function(theta, X)
	D1 = np.zeros(shape=(len(sig), len(sig)))
	D2 = np.zeros(shape=(len(sig), len(sig)))
	for i in range(0, len(sig)):
		D1[i][i] = sig[i]
		D2[i][i] = 1 - sig[i]
	D = np.matmul(D1, D2)
	Xt = np.transpose(X)
	return np.matmul(np.matmul(Xt, D), X)

def singleDerivative(X, theta, Y):
	Xt = np.transpose(X)
	A = sigmoid_function(theta, X)
	return np.matmul(Xt, np.subtract(A, Y))

def NewtonsMethod(X, Y, theta, alpha, iterations):
	t = 0
	thetaprev = np.array([0., 0., 0.])
	for i in range(0, iterations):
		H = HessianMatrix(X, theta)
		theta = theta - np.matmul(np.linalg.inv(H), singleDerivative(X, theta, Y))
		t = t+1
		if(isConversed(theta, thetaprev)):
			return theta, t
		thetaprev[:] = theta[:]
	print("Forced Return")
	return (theta, t)

args = sys.argv
filename1 = "logisticX.csv"
filename2 = "logisticY.csv"
data = genfromtxt(args[1]+"/"+filename1, delimiter=',')
y = genfromtxt(args[1]+"/"+filename2, delimiter=',')
mean1 = np.mean(data[:, 0])
mean2 = np.mean(data[:, 1])
data[:, 0] = np.subtract(data[:, 0], mean1)
data[:, 1] = np.subtract(data[:, 1], mean2)
std1 = np.std(data[:, 0])
std2 = np.std(data[:, 1])
data[:, 0] = np.divide(data[:, 0], std1)
data[:, 1] = np.divide(data[:, 1], std2)

# data = np.subtract(data, np.mean(data))
# data = np.divide(data, np.std(data))
# y = np.loadtxt(filename2)
X = np.ones(shape=(len(y), 3))
X[:, 1] = data[:, 0]
X[:, 2] = data[:, 1]
theta = np.array([0., 0., 0.])
alpha = 0.01
iterations = 5000

x1 = []
y1=[]
x2=[]
y2=[]
for i in range(0, len(y)):
	if(y[i] == 0):
		x1.append(data[i][0])
		y1.append(data[i][1])
	else :
		x2.append(data[i][0])
		y2.append(data[i][1])

# plt.scatter(x1, y1, c='black')
# plt.scatter(x2, y2, c='red')
# plt.show()

L = log_likelihood(X, y, theta)
print("Initial log-likelihood : ", L)
(theta, t) = NewtonsMethod(X, y, theta, alpha, iterations)
# theta = alltheta[count-1]
L = log_likelihood(X, y, theta)
print("Final log-likelihood : ", L)
print("Total iterations : ", t)
print("Final theta : ", theta)

print("Part A completed")
print("--------------------------")
# print("Part B running")

# plt.scatter(x1, y1, c='black')
# plt.scatter(x2, y2, c='red')

# y_plot = findx2(theta, X[:, 1])
# plt.plot(X[:, 1], y_plot)
# plt.savefig(args[2]+"/prediction.png")
# plt.clf()

# print("Part B completed")
# print("--------------------------")