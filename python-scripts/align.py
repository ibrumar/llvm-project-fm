
import sys
import pickle 

print('loading:',sys.argv[1],sys.argv[2],sys.argv[3])

with open(sys.argv[1],'rb') as f:
 f1,f2 = pickle.load(f)

for bb in f1:
  if bb['id']==sys.argv[2]:
    print(bb)

for bb in f2:
  if bb['id']==sys.argv[3]:
    print(bb)

