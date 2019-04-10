import numpy as np 

def main():
	arr = np.array([3,3,3,2])
	idxs = np.argwhere(arr == np.amax(arr))
	final_idx = idxs[len(idxs)-1]
	print(final_idx)

if __name__ == "__main__":
    main()