import os
import sys

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("make directory!")
    except OSError:
        print ('Error: Creating directory. ' +  directory)

#sys.argv[1]

start = int(sys.argv[1])

n =  int(sys.argv[2])


for i in range(start, (start+n)):
    dir1= 'C:/Users/qudwn/Desktop/P'
    dir2= str(i)
    dir = dir1+dir2
    createFolder(dir)

