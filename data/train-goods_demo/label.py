
import numpy as np
import os

##@@ Put this script in the folder of good peak images. @@##

os.system("ls *.png | sed -e 's/\..*$//' > tmp1 ")
os.system("sort -k 1 -g tmp1 > tmp2.txt")

list = np.loadtxt('tmp2.txt')
label = []

for i in range(4546):    ## The range may change according to the number of total samples. ##
    if i in list:
        label.append([i, 1])
    else:
        label.append([i, 0])

#print (np.sum(label, axis = 0))
f1 = ("../label-train.txt")
np.savetxt(f1, label, fmt='%.0f',delimiter='   ')


os.system("rm tmp*")
