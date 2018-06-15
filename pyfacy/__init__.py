import sys,os
root = os.path.dirname(__file__)
root_fr = os.path.dirname(root)

if sys.path[1] != root :
	sys.path.insert(1, root)
if sys.path[2] != root_fr:
	sys.path.insert(1, root_fr)

from utils import *
