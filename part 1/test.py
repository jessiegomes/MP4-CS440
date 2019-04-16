import numpy as np 

def main():
	arr = np.array([3,3,3,2])
	# idxs = np.argwhere(arr == np.amax(arr))
	# final_idx = idxs[len(idxs)-1]
	arr = arr[::-1]
	action = np.argmax(arr)
	action = 3 - action
	print(action)
	(new_x, new_y) = (5, 5)
	body = [(3,3), (4,4), (5,5), (6,6)]
	if (new_x, new_y) in body:
		print("in body")
	else:
		print("not in body")

if __name__ == "__main__":
    main()