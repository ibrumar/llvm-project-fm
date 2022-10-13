
import sys
import pickle 

def loadFunc(filename):
  with open(filename, 'r') as f:
    blockId = None
    block = None
    blocks = []
    for line in f:
      line = line.strip()
      if len(line)>0:
        if blockId==None:
          blockId = line
          block = {}
          block['id'] = blockId
          block['insts'] = []
        else:
          tmp = line.split()
          opcode = tmp[0]
          types = tmp[1:]
          block['insts'].append( {'opcode':opcode, 'types':types} )
      else:
        if blockId!=None:
          blocks.append(block)
          print(block)
          blockId = None
          block = None
  return blocks


f1 = loadFunc(sys.argv[1])
f2 = loadFunc(sys.argv[2])

pairf = (f1,f2)

with open(sys.argv[3],'wb') as f:
 pickle.dump(pairf,f)

