import sys
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math 
from random import seed
from random import gauss
from numpy import genfromtxt

def computeCost(X, y, theta):
	m = len(y)
	H = np.subtract(np.matmul(X, theta), y)
	Hdash = np.transpose(H)
	return (1/(2*len(y)))*np.matmul(Hdash, H)

def isConversed(alltheta, t, X, Y):
	X = np.abs(np.subtract(alltheta[len(alltheta)-1], alltheta[len(alltheta)-2]))
	if(np.sum(X) < 0.001):
		cost1 = computeCost(X, Y, alltheta[len(alltheta)-1])
		cost2 = computeCost(X, Y, alltheta[len(alltheta)-2])
		if(abs(cost1 - cost2) < 0.001):
			return True
		else:
			return False
	else:
		return False

def gradientDescent(X, Y, theta, alpha, iterations, r, totalBatches, thresoldTheta, thresoldCost, K):
	alltheta = np.zeros(shape=(max(2000001, totalBatches), 3))
	alltheta[0] = theta
	allcosts = []
	t=1
	epoch = 0
	k = 0
	for i in range(0, iterations-1):
		for b in range(1, totalBatches+1):
			B = X[(b-1)*r:b*r, :]
			y = Y[(b-1)*r:b*r]
			H1 = np.subtract(np.matmul(B, theta), y)
			H1 = np.matmul(np.transpose(H1), B)
			H1 = np.multiply(H1, alpha/len(y))
			theta = np.subtract(theta, np.transpose(H1))
			alltheta[t] = theta
			t = t+1
		epoch = epoch+1

		# check for convergence
		X1 = np.abs(np.subtract(alltheta[t-1], alltheta[t-2]))
		if(np.sum(X1) < thresoldTheta):
			cost1 = computeCost(X, Y, alltheta[t-1])
			cost2 = computeCost(X, Y, alltheta[t-2])
			if(abs(cost1 - cost2) < thresoldCost):
				k=k+1
			else:
				k=0
		else:
			k=0
		if(k >= K):
			return alltheta, epoch, t
		if(t >= max(2000001, totalBatches)):
			return alltheta, epoch, t
	return alltheta, epoch, t



args = sys.argv
m = 1000000	#number of training sets
print("part A running")
X = np.ones(shape=(3, m))
seed(1)
for i in range(0, m):
	X[1][i] = gauss(3, 2)
	X[2][i] = gauss(-1, 2)

X = np.transpose(X)
theta = np.array([3., 1., 2.])
theta = np.transpose(theta)

Y = np.matmul(X, theta)
for i in range(0, m):
	Y[i] = Y[i]+gauss(0, math.sqrt(2))

# output = np.ones(shape=(m, 2))
# output[:, 0] = X[:, 1]
# output[:, 1] = X[:, 2]
np.savetxt(args[2]+"/q2aSampleX.csv", X[:, [1,2]], delimiter=",")
np.savetxt(args[2]+"/q2aSampleY.csv", Y, delimiter="")
print("Part A completed")
print("--------------------------------")
print("part B running")
iterations = 500
alpha = 0.1


theta1 = theta.copy()
theta1[:] = 0
r = 1	# batch Size
totalBatches = int(m/r)
cost = computeCost(X, Y, theta1)
print(f"For Batch size: ", r)
print(f"Initial cost : {cost}")
print(f"Learning Rate : {alpha/1000}")
alltheta1, epoch1, t1 = gradientDescent(X, Y, theta1, alpha/1000, iterations, r, totalBatches, 0.1, 0.1, 1)
print("total num of times, theta parameters computed : ", t1-1)
print("epoch : ", epoch1)
theta1[:] = alltheta1[t1-1, :]
print("Final theta obtained: ", theta1)
cost1 = computeCost(X, Y, theta1)
print(f"Final cost: {cost1}")
print("----------------------")


theta2 = theta.copy()
theta2[:] = 0
r = 100	# batch Size
totalBatches = int(m/r)
cost = computeCost(X, Y, theta2)
print(f"For Batch size: ", r)
print(f"Initial cost : {cost}")
print(f"Learning Rate : {alpha/10}")
alltheta2, epoch2, t2 = gradientDescent(X, Y, theta2, alpha/10, iterations, r, totalBatches, 0.1, 0.1, 5)
print("total num of times, theta parameters computed : ", t2-1)
print("epoch : ", epoch2)
theta2[:] = alltheta2[t2-1, :]
print("Final theta obtained: ", theta2)
cost2 = computeCost(X, Y, theta2)
print(f"Final cost: {cost2}")
print("----------------------")


theta3 = theta.copy()
theta3[:] = 0
r = 10000	# batch Size
totalBatches = int(m/r)
cost = computeCost(X, Y, theta3)
print(f"For Batch size: ", r)
print(f"Initial cost : {cost}")
print(f"Learning Rate : {alpha}")
alltheta3, epoch3, t3 = gradientDescent(X, Y, theta3, alpha/10, iterations, r, totalBatches, 0.1, 0.1, 6)
print("total num of times, theta parameters computed : ", t3-1)
print("epoch : ", epoch3)
theta3[:] = alltheta3[t3-1, :]
print("Final theta obtained: ", theta3)
cost3 = computeCost(X, Y, theta3)
print(f"Final cost: {cost3}")
print("----------------------")

theta4 = theta.copy()
theta4[:] = 0
r = 1000000	# batch Size
totalBatches = int(m/r)
cost = computeCost(X, Y, theta4)
print(f"For Batch size: ", r)
print(f"Initial cost : {cost}")
print(f"Learning Rate : {alpha/10}")
alltheta4, epoch4, t4 = gradientDescent(X, Y, theta4, alpha/10, iterations, r, totalBatches, 0.01, 0.01, 50)
print("total num of times, theta parameters computed : ", t4-1)
print("epoch : ", epoch4)
theta4[:] = alltheta4[t4-1, :]
print("Final theta obtained: ", theta4)
cost4 = computeCost(X, Y, theta4)
print(f"Final cost: {cost4}")


print("Part B completed")
print("--------------------------------")
print("Part C running")
test = genfromtxt(args[1]+"/"+"q2test.csv", delimiter=',')
test = np.delete(test, 0, 0)
data = np.ones(shape=(test.shape[0], 3))
data[:, 1] = test[:, 0]
data[:, 2] = test[:, 1]
Y = test[:, 2]
cost1 = computeCost(data, Y, theta1)
cost2 = computeCost(data, Y, theta2)
cost3 = computeCost(data, Y, theta3)
cost4 = computeCost(data, Y, theta4)
cost5 = computeCost(data, Y, theta)
print("cost for sample when batchSize is 1 : ", cost1)
print("cost for sample when batchSize is 100 : ", cost2)
print("cost for sample when batchSize is 10000 : ", cost3)
print("cost for sample when batchSize is 1000000 : ", cost4)
print("")
print("cost for sample with original theta : ", cost5)
print("Part C completed")
print("--------------------------------")



print("Part D running")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print("running for batchSize = 1:")
ax.set_zlim([int(min(alltheta1[:, 0]))-1, int(max(alltheta1[:, 0]))+1])
ax.set_xlim([int(min(alltheta1[:, 1]))-1, int(max(alltheta1[:, 1]))+1])
ax.set_ylim([int(min(alltheta1[:, 2]))-1, int(max(alltheta1[:, 2]))+1])
ax.plot(alltheta1[:,0][0:t1-1], alltheta1[:,1][0:t1-1], alltheta1[:,2][0:t1-1])
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('theta_2')
plt.title('theta movement for batch size : 1')
plt.savefig(args[2]+"/3dtheta1.png")
plt.close()
print("plot saved")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print("running for batchSize = 100:")
ax.set_zlim([int(min(alltheta2[:, 0]))-1, int(max(alltheta2[:, 0]))+1])
ax.set_xlim([int(min(alltheta2[:, 1]))-1, int(max(alltheta2[:, 1]))+1])
ax.set_ylim([int(min(alltheta2[:, 2]))-1, int(max(alltheta2[:, 2]))+1])
ax.plot(alltheta2[:,0][0:t2-1], alltheta2[:,1][0:t2-1], alltheta2[:,2][0:t2-1])
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('theta_2')
plt.title('theta movement for batch size : 100')
plt.savefig(args[2]+"/3dtheta2.png")
plt.close()
print("plot saved")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print("running for batchSize = 10000:")
ax.set_zlim([int(min(alltheta3[:, 0]))-1, int(max(alltheta3[:, 0]))+1])
ax.set_xlim([int(min(alltheta3[:, 1]))-1, int(max(alltheta3[:, 1]))+1])
ax.set_ylim([int(min(alltheta3[:, 2]))-1, int(max(alltheta3[:, 2]))+1])
ax.plot(alltheta3[:,0][0:t3-1], alltheta3[:,1][0:t3-1], alltheta3[:,2][0:t3-1])
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('theta_2')
plt.title('theta movement for batch size : 10000')
plt.savefig(args[2]+"/3dtheta3.png")
plt.close()
print("plot saved")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print("running for batchSize = 1000000:")
ax.set_zlim([int(min(alltheta4[:, 0]))-1, int(max(alltheta4[:, 0]))+1])
ax.set_xlim([int(min(alltheta4[:, 1]))-1, int(max(alltheta4[:, 1]))+1])
ax.set_ylim([int(min(alltheta4[:, 2]))-1, int(max(alltheta4[:, 2]))+1])
ax.plot(alltheta4[:,0][0:t4-1], alltheta4[:,1][0:t4-1], alltheta4[:,2][0:t4-1])
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('theta_2')
plt.title('theta movement for batch size : 1000000')
plt.savefig(args[2]+"/3dtheta4.png")
plt.close()
print("plot saved")

print("Part D completed")