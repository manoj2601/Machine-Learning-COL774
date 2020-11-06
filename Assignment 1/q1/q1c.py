import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d 

"""
Computes the cost for given X, y and theta
J(theta) = 1/2m * sum_from i = 0 to m {(y - theta'.x)^2}
"""
def computeCost(X, y, theta):
	m = len(y)
	H = np.subtract(np.matmul(X, theta), y)
	Hdash = np.transpose(H)
	return (1/(2*len(y)))*np.matmul(Hdash, H)

"""
Converging criteria
"""
def isConversed(theta1, theta2, cost1, cost2):
	X = np.abs(np.subtract(theta2, theta1))
	Y = np.abs(cost1 - cost2)
	if(np.sum(X) < 0.0000001 and Y < 0.0000001):
		return True
	return False

"""
Computes cost for theta0 and theta1
"""
def CalculateJ(theta0, theta1, X, y):
	thetaa = [theta0, theta1]
	return computeCost(X, y, thetaa)
	
def gradientDescent(X, y, theta, alpha, iterations):
	alltheta = np.zeros((iterations+1, 2), float)
	alltheta[0, :] = theta
	allcosts = np.zeros(iterations+1, float)
	allcosts[0] = computeCost(X, y, theta)
	epoch = 1
	while(epoch <= iterations):
		H1 = np.subtract(np.matmul(X, theta), y)
		H1 = np.matmul(np.transpose(H1), X)
		H1 = np.multiply(H1, alpha/len(y))
		theta = np.subtract(theta, np.transpose(H1))
		alltheta[epoch, :] = theta[:]
		allcosts[epoch] = computeCost(X, y, theta)

		if(isConversed(theta, alltheta[epoch-1], allcosts[epoch], allcosts[epoch-1])):
			return alltheta, allcosts, epoch		
		epoch = epoch + 1
	return alltheta, allcosts, epoch

filename1 = "linearX.csv"
filename2 = "linearY.csv"
total_argu = len(sys.argv)
args = sys.argv
data = np.loadtxt(args[1]+"/linearX.csv")
data = np.subtract(data, np.mean(data))
data = np.divide(data, np.std(data))
y = np.loadtxt(args[1]+"/linearY.csv")
X = np.ones(shape=(2,len(y)))
X[1, :] = data[:]
X = np.transpose(X)
theta = [0, 0]

iterations = 50000
alpha = 0.001
cost = computeCost(X, y, theta)
alltheta, allcosts, count = gradientDescent(X, y, theta, alpha, iterations)
theta[:] = alltheta[count-1, :]
cost = computeCost(X, y, theta)
x = alltheta[:, 0]
y1 = alltheta[:, 1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim([0, max(allcosts)])
ax.set_xlim([min(x), max(x)])
ax.set_ylim([min(y1), max(y1)])
ax.set_xlabel('$theta_0$')
ax.set_ylabel('$theta_1$')
ax.set_zlabel('$cost : J(theta)$')

ax.scatter(x[0:count], y1[0:count], allcosts[0:count], label='converging values of cost (J(theta))')
ax.legend(loc="upper right")
plt.title("3-dimensional mesh showing the error function (J(θ))")
plt.savefig(args[2]+"/3dmesh.png")
# plt.show()
plt.clf()

"""d part"""

# x1 = np.linspace(-0.2, 2, 100)
# x2 = np.linspace(-0.6, 0.6, 100)
# X1, X2 = np.meshgrid(x1, x2)
# Z = np.zeros(shape=(X1.shape))
# for i in range(0, 100):
# 	for j in range(0, 100):
# 		Z[i, j] = CalculateJ(X1[i,j], X2[i,j], X, y)
# plt.contour(X1, X2, Z, levels = 20)
# plt.savefig(args[2]+"/contour.png")
# plt.clf()