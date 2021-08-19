import datetime

def wlog(file, text, flag = 0):
	""" This function writes to log file the recieved text
	
	Input Data:
		file - the log file path. Datatype: string
		text - the text to be written in the file log. Datatye: string
		flag - values 0 or 1. If value is 1, the data and time will pe written before text. DEFAULT = 0
		
	"""
	current_time = datetime.datetime.now() 
	current_time = current_time.strftime("%d-%b-%Y (%H:%M:%S)")

	file = open(file,'a')
	file.write("\n")

	if flag==1:
		file.write(current_time)
		file.write("\n")
	file.writelines(text)
	file.write("\n\n")
	file.close()