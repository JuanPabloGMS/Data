import pandas as pd
import numpy as np
df = []
df_sep, df_oct, df_only_oct, df_onlyA_both = [],[],[],[]
def group1():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	#Filtrar data frame por el mes de octubre con condicional
	cond_oct = df["CORTE ARCHIVO"] == "octubre"
	cond_sep = df["CORTE ARCHIVO"] == "septiembre"
	df_sep = df[cond_sep]
	df_oct = df[cond_oct]
	creds_sep = set(df_sep["CREDITO"])
	df_only_oct = df_oct[~df_oct.CREDITO.isin(creds_sep)]
	df_only_oct.shape
	return df_only_oct

def group11():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check = df_only_oct["SALDO"] == df_only_oct["MONTO"]
	df_g11 = df_only_oct[check]
	return df_g11

def group12():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check = df_only_oct["SALDO"] < df_only_oct["MONTO"]
	df_g12 = df_only_oct[check]
	return df_g12

def group13():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check = df_only_oct["SALDO"] > df_only_oct["MONTO"]
	df_g13 = df_only_oct[check]
	return df_g13

def group2():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	creds_oct = set(df_oct["CREDITO"])
	df_only_sep = df_sep[~df_sep.CREDITO.isin(creds_oct)]
	return df_only_sep

def group3():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check_K = df_oct['ESTADO'] == "K"
	df_K = df_oct[check_K]
	return df_K

def group41():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check_Koct = df_oct['ESTADO'] == "A"
	check_Ksep = df_sep['ESTADO'] == "A"
	df_onlyA_sep = df_sep[check_Ksep]
	df_onlyA_oct = df_oct[check_Koct]
	df_onlyA_both = pd.merge(df_onlyA_sep, df_onlyA_oct, how = 'inner', on=["CREDITO"])
	check_gtsaldo = df_onlyA_both["SALDO_x"] > df_onlyA_both["SALDO_y"]
	check_eqmonto = df_onlyA_both["MONTO_x"] == df_onlyA_both["MONTO_y"]
	df_41 = df_onlyA_both[check_gtsaldo & check_eqmonto]
	return df_41
def group42():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check_gtsaldo_eqmonto = (df_onlyA_both["SALDO_x"] > df_onlyA_both["SALDO_y"]) & (df_onlyA_both["MONTO_x"] < df_onlyA_both["MONTO_y"])
	df_42 = df_onlyA_both[check_gtsaldo_eqmonto]
	return df_42

def group43():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check_gtsaldo_eqmonto = (df_onlyA_both["SALDO_x"] > df_onlyA_both["SALDO_y"]) & (df_onlyA_both["MONTO_x"] > df_onlyA_both["MONTO_y"])
	df_43 = df_onlyA_both[check_gtsaldo_eqmonto]
	return df_43

def group44():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check_gtsaldo_eqmonto = (df_onlyA_both["SALDO_x"] < df_onlyA_both["SALDO_y"]) & (df_onlyA_both["MONTO_x"] == df_onlyA_both["MONTO_y"])
	df_44 = df_onlyA_both[check_gtsaldo_eqmonto]
	return df_44

def group45():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check_gtsaldo_eqmonto = (df_onlyA_both["SALDO_x"] < df_onlyA_both["SALDO_y"]) & (df_onlyA_both["MONTO_x"] < df_onlyA_both["MONTO_y"])
	df_45 = df_onlyA_both[check_gtsaldo_eqmonto]
	return df_45

def group46():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check_gtsaldo_eqmonto = (df_onlyA_both["SALDO_x"] < df_onlyA_both["SALDO_y"]) & (df_onlyA_both["MONTO_x"] > df_onlyA_both["MONTO_y"])
	df_46 = df_onlyA_both[check_gtsaldo_eqmonto]
	return df_46
def group47():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check_gtsaldo_eqmonto = (df_onlyA_both["SALDO_x"] == df_onlyA_both["SALDO_y"]) & (df_onlyA_both["MONTO_x"] == df_onlyA_both["MONTO_y"])
	df_47 = df_onlyA_both[check_gtsaldo_eqmonto]
	return df_47
def group48():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check_gtsaldo_eqmonto = (df_onlyA_both["SALDO_x"] == df_onlyA_both["SALDO_y"]) & (df_onlyA_both["MONTO_x"] < df_onlyA_both["MONTO_y"])
	df_48 = df_onlyA_both[check_gtsaldo_eqmonto]
	return df_48
def group49():
	global df, df_sep, df_oct, df_only_oct, df_onlyA_both
	check_gtsaldo_eqmonto = (df_onlyA_both["SALDO_x"] == df_onlyA_both["SALDO_y"]) & (df_onlyA_both["MONTO_x"] > df_onlyA_both["MONTO_y"])
	df_49 = df_onlyA_both[check_gtsaldo_eqmonto]
	return df_49

def main():
	global df
	df = pd.read_excel("database.xlsx",sheet_name = 'ORIGINAL')
	with pd.ExcelWriter('output.xlsx') as writer:
		group1().to_excel(writer, sheet_name = 'S_1')
		group11().to_excel(writer, sheet_name = 'S_1.1')
		group12().to_excel(writer, sheet_name = 'S_1.2')
		group13().to_excel(writer, sheet_name = 'S_1.3')
		group2().to_excel(writer, sheet_name = 'S_2')
		group3().to_excel(writer, sheet_name = 'S_3')
		group41().to_excel(writer, sheet_name = 'S_4.1')
		group42().to_excel(writer, sheet_name = 'S_4.2')
		group43().to_excel(writer, sheet_name = 'S_4.3')
		group44().to_excel(writer, sheet_name = 'S_4.4')
		group45().to_excel(writer, sheet_name = 'S_4.5')
		group46().to_excel(writer, sheet_name = 'S_4.6')
		group47().to_excel(writer, sheet_name = 'S_4.7')
		group48().to_excel(writer, sheet_name = 'S_4.8')
		group49().to_excel(writer, sheet_name = 'S_4.9')
main()