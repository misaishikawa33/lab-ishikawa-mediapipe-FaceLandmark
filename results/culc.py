# 変動計算

import pandas as pd
import numpy as np
import sys
import glob
import os

def culc_coef(dirname, filename):
    dirpath = "./" + dirname + "/"
    filelist = glob.glob(os.path.join(dirpath, "*.dat"))
    output = open(filename, mode='w')
    for filepath in filelist:
        df = pd.read_csv(filepath, header=None, names=['x', 'y'])
        a = df.values
        xmean = np.mean(a, axis=0)[0]
        ymean = np.mean(a, axis=0)[1]
        xstd = np.std(a, axis=0)[0]
        ystd = np.std(a, axis=0)[1]
        output.write(str(xstd/xmean)+","+str(ystd/ymean)+"\n")
    output.close()
    
def culc_error(filename1, filename2, outfilename):
    f1 = os.path.join("error_facemesh",filename1)
    f2 = os.path.join("error_facemesh",filename2)
    df1 = pd.read_csv(f1, header=None, usecols = [1,2], names=['x', 'y'])
    df2 = pd.read_csv(f2, header=None, usecols = [1,2], names=['x', 'y'])
    if len(df1) != len(df2):
        print("要素数が異なります")
        sys.exit()
    a1 = df1.values
    a2 = df2.values
    print(a1)
    a3 = []
    
    for i in range(df1.shape[0]):
        dist = np.linalg.norm(a1[i]-a2[i])
        a3.append(dist)
        # output.write(str(a1[i][0]-a2[i][0])+","+str(a1[i][1]-a2[i][1])+","+str(a1[i][2]-a2[i][2])+"\n")
    # x = sorted(a3, key=lambda x: x[1])
    # y = sorted(a3, key=lambda x: x[2])
    with open("error_facemesh/error_facemesh.txt","a") as f:
        np.savetxt(f, a3,fmt="%.4f")
    

if __name__ == '__main__': 
    print("計算方法を指定してください：")
    mode = input("変動係数=c, ランドマーク間のマスク着用/非着用時の誤差=e, 詳しい説明=h")
    if mode == "c":
        culc_coef(sys.argv[1], sys.argv[2])
    elif mode == "e":
        culc_error(sys.argv[1], sys.argv[2], sys.argv[3])
            
        