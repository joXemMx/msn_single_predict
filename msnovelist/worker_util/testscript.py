from predict import MSNovelist
import numpy as np

fp = np.loadtxt("/home/joxem/worker-msnovelist/worker_util/laudanosine_fp.txt", delimiter=',')
mf = 'C21H27NO4'
result = MSNovelist.predict(mf, fp)
print(result)