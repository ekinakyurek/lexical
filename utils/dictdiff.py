import sys
from collections import Counter
import pdb
import json
import operator

def main(d1, d2):
	with open(d1, "r") as f:
		v1 = json.load(f)
		for (k,v) in v1.items():
			max_k = max(v.items(), key=operator.itemgetter(1))[0]
			v1[k] = max_k


	with open(d2, "r") as f:
		v2 = json.load(f)
		for (k,v) in v2.items():
			max_k = max(v.items(), key=operator.itemgetter(1))[0]
			v2[k] = max_k

	for (k,v) in v1.items():
		if k in v2 and v2[k] != v:
			print("v1: ", k, " -> " , v, " where v2: ", k, " -> ", v2[k])
		elif k not in v2:
			print("v1: ", k, " -> " , v, " where v2")

	for	(k,v) in v2.items():
		if k in	v1 and v1[k] !=	v:
			print("v2: ", k, " -> " , v, " where v1: ", k, " -> ", v1[k])
		elif k not in v2:
			print("v2: ", k, " -> " , v, " where v1")



if __name__ == "__main__":
	# execute only if run as a script
	assert len(sys.argv) > 2
	main(sys.argv[1], sys.argv[2])
