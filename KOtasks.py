import numpy as np

def task(y, str = 'bilabial'):
	"""
	This function transform the 11 classes of Kara One database into the specific tasks

	Input Data:
		y - the taget havin the 11 classes
		str - str can be: 'bilabial', 'nasal', 'C/V', 'iy', 'uw'. These are the tasks used for classification

			'bilabial' = piy, pat, pot, m
			'nasal' = m, n, knew, gnaw
			'C/V' = piy, tiy, diy, m, n
			'iy' = iy
			'uw' = uw

			The targets are labeld 1, in rest 0

			labels:
				0 - iy
				1 - uw
				2 - piy
				3 - tiy
				4 - diy
				5 - m
				6 - n
				7 - pat
				8 - pot
				9 - knew
				10 - gnaw

	Output Data:
		yt - the target modified regarding the task
	"""

	yt = np.zeros(y.shape)

	if str == 'bilabial':
		yt[y==2] = 1
		yt[y==5] = 1
		yt[y==7] = 1
		yt[y==8] = 1
	
	elif str=='nasal':
		yt[y==5] = 1
		yt[y==6] = 1
		yt[y==9] = 1
		yt[y==10] = 1

	elif str=='C/V':
		yt[y==2] = 1
		yt[y==3] = 1
		yt[y==4] = 1
		yt[y==5] = 1
		yt[y==6] = 1

	elif str=='iy':
		yt[y==0] = 1

	elif str=='uw':
		yt[y==1] = 1

	else:
		raise ValueError("It's not a valid task")

	return yt