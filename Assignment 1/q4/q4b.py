import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
from numpy import genfromtxt
import math


def findY(constant1, constant2, coeffx, coeffy, x):
	return ((constant1 - constant2 - x*coeffx)/coeffy)

args = sys.argv
filename1 = "q4x.dat"
filename2 = "q4y.dat"
data = genfromtxt(args[1]+"/"+filename1, delimiter='')
# y = genfromtxt(args[1]+"/"+filename2, delimiter='')
f = open(args[1]+"/"+filename2, "r")
y = []
ones = 0
zeros=0
while(True):
	a = f.readline()
	if(a == "Alaska\n"): # 0 means Alaska
		zeros=zeros+1
		y.append(0)
	elif(a == "Canada\n"): # 1 means Canada
		ones=ones+1
		y.append(1)
	else:
		break

mean1 = np.mean(data[:, 0])
mean2 = np.mean(data[:, 1])
data[:, 0] = np.subtract(data[:, 0], mean1)
data[:, 1] = np.subtract(data[:, 1], mean2)
std1 = np.std(data[:, 0])
std2 = np.std(data[:, 1])
data[:, 0] = np.divide(data[:, 0], std1)
data[:, 1] = np.divide(data[:, 1], std2)

fig = plt.figure()
m = len(y)
class1 = []
class2 = []
for i in range(0, len(y)):
	if(y[i] == 0):
		class1.append([data[i][0], data[i][1]])
	else:
		class2.append([data[i][0], data[i][1]])
print("Part A running")
class1 = np.array(class1)
class2 = np.array(class2)
phi = np.sum(y)/m

u01 = np.sum(class1[:, 0])/len(class1)
u02 = np.sum(class1[:, 1])/len(class1)
u0 = np.array([u01, u02])

u11 = np.sum(class2[:, 0])/len(class2)
u12 = np.sum(class2[:, 1])/len(class2)
u1 = np.array([u11, u12])

X = data.copy()

for i in range(0, m):
	if(y[i] == 0):
		X[i][0] = X[i][0] - u0[0]
		X[i][1] = X[i][1] - u0[1]
	elif(y[i] == 1):
		X[i][0] = X[i][0] - u1[0]
		X[i][1] = X[i][1] - u1[1]

sigma = np.matmul(np.transpose(X), X)


print("phi : ", phi)
print("mu0 : ", u0)
print("mu1 : ", u1)
print("covariance Matrix : \n", sigma)

print("Part A completed")
print("-------------------------")
print("Part B running")
plt.scatter(class1[:, 0], class1[:,1], c='black', label="Alaska")
plt.scatter(class2[:, 0], class2[:,1], c='red', label="Canada")
plt.title('Normalized data (black: Alaska; red: Canada)')
plt.xlabel('normalized growth ring diameter in fresh water')
plt.ylabel('normalized growth ring diameter in marine water')
plt.legend()
plt.savefig(args[2]+"/q4b.png")
plt.close()
print("Part B completed")


# print("Part C running")

# siginv = np.linalg.inv(sigma)
# a1 = siginv[0][0]
# a2 = siginv[0][1]
# a3 = siginv[1][0]
# a4 = siginv[1][1]

# constant1 = (-2)*math.log(phi/(1-phi))

# constant2 = a1*(u1[0]**2 - u0[0]**2) +(a2+a3)*(u1[0]*u1[1]-u0[1]*u0[0]) + a4*(u1[1]**2 - u0[1]**2)

# coeffx = 2*a1*(u0[0] - u1[0]) + (a2+a3)*(u0[1] - u1[1])
# coeffy = 2*a4*(u0[1]-u1[1]) + (a2+a3)*(u0[0]-u1[0])

# y_arr = []
# for i in range(0, len(y)):
# 	y_arr.append(findY(constant1, constant2, coeffx, coeffy, data[i][0]))


# # plt.scatter(class1[:, 0], class1[:,1], c='black')
# # plt.scatter(class2[:, 0], class2[:,1], c='red')
# # plt.title('Normalized data (black: Alaska; red: Canada)')
# # plt.xlabel('normalized growth ring diameter in fresh water')
# # plt.ylabel('normalized growth ring diameter in marine water')
# # plt.plot(data[:, 0], y_arr)
# # plt.show()
# # plt.clf()
# print("Part C completed")
# print("-------------------------")

# print("Part D running")

# copy1 = class1.copy()
# for i in range(0, len(copy1)):
# 	copy1[i][0] = copy1[i][0] - u0[0]
# 	copy1[i][1] = copy1[i][1] - u0[1]

# copy1t = np.transpose(copy1)
# sig0 = np.matmul(copy1t, copy1)

# sig0 = np.true_divide(sig0, len(class1))


# copy2 = class2.copy()
# for i in range(0, len(copy2)):
# 	copy2[i][0] = copy2[i][0] - u1[0]
# 	copy2[i][1] = copy2[i][1] - u1[1]

# copy2t = np.transpose(copy2)
# sig1 = np.matmul(copy2t, copy2)

# sig1 = np.true_divide(sig1, len(class2))

# print("u0 : ", u0)
# print("u1 : ", u1)
# print("sigma0 : ", sig0)
# print("sigma1 : ", sig1)
# print("Part D completed")
# print("-------------------------")


# print("Part E running")

# sig0inv = np.linalg.inv(sig0)
# sig1inv = np.linalg.inv(sig1)

# x0 = np.outer(np.linspace(min(data[:,0]),max(data[:,0]),30),np.ones(30))
# x1 = np.outer(np.ones(30),np.linspace(min(data[:,1]),max(data[:,1]),30))
# z = np.outer(np.ones(30),np.ones(30))
# for i in range(30):
#     for j in range(30):
#         x = [x0[i][j],x1[i][j]]
#         a = np.matmul(x, np.matmul(np.subtract(sig0inv, sig1inv), np.transpose(x)))
#         b = np.multiply(np.matmul(x, np.subtract(np.matmul(sig1inv, u1), np.matmul(sig0inv, u0))), 2)
#         c = np.subtract(np.matmul(np.transpose(u0), np.matmul(sig0inv, u0)), np.matmul(np.transpose(u1), np.matmul(sig1inv, u1)))
#         d = 2*math.log(phi/(1-phi))
#         e = math.log(np.linalg.det(sig0)/np.linalg.det(sig1))
#         z[i][j] = a + b + c + d + e
# plt.contour(x0,x1,z,levels=[0])
# plt.show()