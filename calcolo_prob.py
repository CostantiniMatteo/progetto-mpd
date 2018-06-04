import pandas as pd

#ritorna la sequenza di adls: ['Sleeping', 'Toileting', 'Showering', 'Breakfast', 'Grooming', 'Spare_Time/TV', 'Leaving', 'Lunch', 'Snack']
#ritorna la sequenza di sensors: ['Bed', 'Cabinet', 'Basin', 'Toilet', 'Shower', 'Fridge', 'Cupboard', 'Toaster', 'Cooktop', 'Microwave', 'Seat', 'Maindoor']
def find_el(el_column):
	tmp = []
	for i, el in enumerate(el_column):
		try:
    			tmp.index(el)
		except ValueError:
			tmp.append(el)
	return tmp

#associa ad ogni adl un numero: ['Sleeping'-> 0, 'Toileting'-> 1, 'Showering'-> 2, 'Breakfast'-> 3, 'Grooming'-> 4, 'Spare_Time/TV'-> 5, 'Leaving'-> 6, 'Lunch'-> 7, 'Snack'-> 8]
def conv_adls(adls_column):
	tmp = []
	for i, adl in enumerate(adls_column):
		if adl == 'Sleeping':
			tmp.append(0)
		elif adl == 'Toileting':
			tmp.append(1)
		elif adl == 'Showering':
			tmp.append(2)
		elif adl == 'Breakfast':
			tmp.append(3)
		elif adl == 'Grooming':
			tmp.append(4)
		elif adl == 'Spare_Time/TV':
			tmp.append(5)
		elif adl == 'Leaving':
			tmp.append(6)
		elif adl == 'Lunch':
			tmp.append(7)
		elif adl == 'Snack':
			tmp.append(8)
	return tmp

#associa ad ogni adl un numero: ['Bed'-> 0, 'Cabinet'-> 1, 'Basin'-> 2, 'Toilet'-> 3, 'Shower'-> 4, 'Fridge'-> 5, 'Cupboard'-> 6, 'Toaster'-> 7, 'Cooktop'-> 8, 'Microwave'-> 9, 'Seat'-> 10, 'Maindoor'-> 11 ]
def conv_sens(sens_column):
	tmp = []
	for i, sen in enumerate(sens_column):
		if sen == 'Bed':
			tmp.append(0)
		elif sen == 'Cabinet':
			tmp.append(1)
		elif sen == 'Basin':
			tmp.append(2)
		elif sen == 'Toilet':
			tmp.append(3)
		elif sen == 'Shower':
			tmp.append(4)
		elif sen == 'Fridge':
			tmp.append(5)
		elif sen == 'Cupboard':
			tmp.append(6)
		elif sen == 'Toaster':
			tmp.append(7)
		elif sen == 'Cooktop':
			tmp.append(8)
		elif sen == 'Microwave':
			tmp.append(9)
		elif sen == 'Seat':
			tmp.append(10)
		elif sen == 'Maindoor':
			tmp.append(11)
	return tmp

#nota: transitions è il risultato di adls_conv
def transition_matrix(transitions):
    n = 1 + max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

#nota: sensors è il risultato di sens_conv
def prior_prob(sensors):
	tmp = [0] * 12
	for i in range(0, len(sensors)):		
		tmp[sensors[i]] += 1
	for i in range(0, len(tmp)):		
		tmp[i] = tmp[i] / len(sensors)
	return tmp
	


df = pd.read_csv('OrdonezA_Sensors.csv')
saved_column = df['3']##modifica con nome colonna
x = prior_prob(conv_sens(saved_column))
print("Probabilità a priori:")
print (x)
tot = 0
for i in range(0, len(x)):		
	tot += x[i]
print()
print ("Somma delle probabilità a priori:")
print (tot)


