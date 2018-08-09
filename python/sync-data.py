import pandas as pd
import numpy as np

for i in range(1,24):  
    path= '../data/advio-'+str(i).zfill(2)+'/iphone/arkit.csv'
    arkit=pd.read_csv(path,names=list('tabcdefg'))
    path= '../data/advio-'+str(i).zfill(2)+'/iphone/accelerometer.csv'
    acc=( pd.read_csv(path,names=list('tabc')))
    path= '../data/advio-'+str(i).zfill(2)+'/iphone/gyro.csv'
    gyro=( pd.read_csv(path,names=list('tabc')))
    g=[]
    a=[]
    t=np.array((map(float,acc[list('t')].values)))
    zer=t*0
    for c in 'abc':
        g.append(np.interp(np.array((map(float,acc[list('t')].values))),np.array((map(float,gyro[list('t')].values))),np.array((map(float,gyro[list(c)].values)))))
        a.append(np.array((map(float,acc[list(c)].values))))
    M=np.column_stack((t,zer+34,g[0],g[1],g[2],a[0],a[1],a[2],zer,zer))
    v=[]
    t=np.array((map(float,arkit[list('t')].values)))
    zer=t*0
    for c in 'abcdefg':
        v.append(np.array((map(float,arkit[list(c)].values))))
    Mkit=np.column_stack((t,zer+7,zer,v[0],v[1],v[2],v[3],v[4],v[5],v[6]))
    full=np.concatenate((M,Mkit))
    full = full[full[:,0].argsort()]
    path= '../data/advio-'+str(i).zfill(2)+'/iphone/imu-gyro.csv'
    np.savetxt(path, full, delimiter=",",fmt='%.7f')