import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np

#================================================================================
#======================================| PROBLEMA |==============================
#================================================================================

# • Problema de la suma de subconjuntos: dado un conjunto de valores
# enteros S y un valor entero M, encontrar un subconjunto de S que
# suma exactamente M.

def generate_random_instance(n_i):
    S = set([random.randint(1,int(n_i*2)) for _ in range(n_i)])
    n = len(S)
    M = sum(random.sample(S,random.randint(1,n-1)))
    return (S,n,M)

S,n,M = generate_random_instance(50)
print("====| Conjunto S |====")
print(S,"Tamaño:",n)
print("====| Entero M |====")
print(M)


#================================================================================
#============================| ALGORÍTMO GENÉTICO |==============================
#================================================================================

# Función de selección
# Elige dos individuos aleatoriamente que van a ser cruzados
# generacion: lista de individuos que componen la generacion actual
# Retorna las posiciones de dos individuos en la generacion
def seleccion(generacion):
    tGen = len(generacion)
    ind1 = random.randint(1, tGen-2)
    ind2 = ind1
    while ind1 == ind2:
        ind2 = random.randint(1,tGen-2)
    return (generacion[ind1], generacion[ind2])

# Funcion de descarte de los individuos menos aptos
# generacion: lista de individuos que componen la generacion actual
# Retorna la generacion despues de eliminar la mitad menos apta
def descarte(generacion):
    tGen = len(generacion)
    return (generacion[:tGen//2])

# Funcion de cruce
# Precondicion: ambos individuos tienen la misma longitud
# ind1 e ind2 son individuos de la generacion actual
#Retorna dos nuevos individuos obtenidos a partir de ind1 e ind2 por cruce
def cruce(ind1,ind2):
    tInd = len(ind1)
    pivot = random.randint(1,max(tInd-1,1))
    new1 = list(set(ind1[:pivot] + ind2[pivot:]))
    new2 = list(set(ind2[:pivot] + ind1[pivot:]))
    return (new1, new2)

# Funcion de mutacion
# ind es un individuo de la generacion actual
# prob es un valor entre 0 y 1 que corresponde a la probabilidad de mutacion
# Retorna un individuo que puede ser identico al que entró o puede tener un cambio aleatorio en una posicion
def mutacion(ind, prob):
    p = random.randint(1,100)
    if p < prob*100: 
        tInd = len(ind)
        q = random.randint(0,tInd-1)
        ind[q] = random.randint(1,20) # esto se debe ajustar de acuerdo a los valores adecuados para el individuo
    return (list(set(ind)))

# Funcion newInd
# Genera un nuevo individuo aleatorio
# Retorna el individuo construido
# Esta funcion debe ajustarse en terminos de la tarea que se vaya a resolver, pues la forma y longitud del individuo varian
def newInd():
    return list(random.sample(S,random.randint(2,n-1)))


# Funcion primeraGen
# nIndGen: numero de individuos por generacion
# Retorna la primera generacion poblada con el numero de individuos requeridos
# Esta funcion depende completamente del problema que se va a resolver, pues el individuo depende del problema
def primeraGen(nIndGen):
    generacion = []
    while len(generacion) < nIndGen:
        generacion.append(newInd())
    return generacion

# Funcion fitness
# ind: es un individuo de la generacion actual
# Retorna un valor numerico que representa la aptitud del individuo
# Esta funcion depende completamente del problema que se va a resolver, pues el puntaje asociado al individuo depende del problema
def fitness(ind):
	return abs(sum(ind)-M)*-1
# Funcion general
# nIndGen: numero de individuos por generacion
# nGen: numero de generaciones que realizara el algoritmo
# pMut: probabilidad de mutacion
def genetico(nIndGen,nGen,pMut):
    generacion = primeraGen(nIndGen)
    while nGen > 0:
        generacion.sort(key = fitness, reverse=True)
        generacion = descarte(generacion)
        children = []
        while len(children) + len(generacion) < nIndGen:
            parent1, parent2 = seleccion(generacion)
            child1, child2 = cruce(parent1,parent2)
            child1 = mutacion(child1, pMut)
            child2 = mutacion(child2, pMut)
            children.append(child1)
            children.append(child2)
        generacion = generacion + children
        nGen = nGen - 1
    return (generacion[0], sum(generacion[0]))

#================================================================================
#=============================| ENFRIAMIENTO SIMULADO |==========================
#================================================================================

# Funcion select_random()
# Genera una variable aleatoria
# Retorna la variable construida
# Esta funcion debe ajustarse en terminos de la tarea que se vaya a resolver, pues la forma y longitud del individuo varian
def select_random():
    return random.sample(S,random.randint(1,n))

# Funcion f(sol)
# sol: posible solucion 
# Calcula la calidad de la solucion dada por parámetro
# Retorna el poderado calculado. 
# Esta funcion debe ajustarse en terminos de la tarea que se vaya a resolver, pues la forma y longitud del individuo varian
def f(sol):
    return abs(sum(sol)-M)

# Funcion d(dt,T)
# dt: diferencia de la calidad entre un par de variables aleatorias 
# T: temperatura actual
# Retorna la probabilidad de escogencia de la nueva variable aleatoria. 
def e(dt,T):
    return math.exp(dt/T)

#Funcion simulated_annealing(T,numIt,cd_factor)
# T: temperatura inicial
# numIt: numero de iteraciones entre cada enfriamiento
# cd_factor: factor enfriamiento
def simulated_annealing(T, numIt, cd_factor):
    initial_sol = select_random();
    actual_sol = initial_sol
    while T > 0:
        for i in range(numIt):
            rand_sol = select_random();
            delta_sol = f(actual_sol) - f(rand_sol)
            if delta_sol > 0:
                actual_sol = rand_sol
            elif random.uniform(0,1) < e(delta_sol,T):
                actual_sol = rand_sol
        T-=cd_factor;
    return (actual_sol, sum(actual_sol))


#================================================================================
#=============================| PROGRAMA PRINCIPAL |=============================
#================================================================================
#Instancia del Problema
print("====| INSTANCIA DEL PROBLEMA |====")
print("Conjunto:",S,"Tamaño:",n)
print("Suma M:",M)
print("-----------------------------------")

def get_exe(n):
    exe_gen = []
    exe_sim = []
    times_gen =  []
    times_sim = []
    for _ in range (n):
        #Procesamiento de Ejecuciones Algoritmo Genético
        t_i = time.time()
        exe_i = genetico(20,200,0.1)
        t = time.time()-t_i
        exe_gen.append(exe_i)
        times_gen.append(t)
        #Procesamiento de Ejecuciones Enfriamiento Simulado
        t_i = time.time()
        exe_i = simulated_annealing(40,100,0.1)
        t = time.time()-t_i
        exe_sim.append(exe_i)
        times_sim.append(t)
    return (exe_gen,exe_sim,times_gen,times_sim)


#n Ejecuciones
n_exe = 50
exeg, exes, time_g, time_s = get_exe(n_exe)
setlen_gen = []
setlen_sim = []
for s,m in exeg:
    if m == M:
        setlen_gen.append(len(s))
for s,m in exes:
    if m == M:
        setlen_sim.append(len(s))
exe_gen = [m for s,m in exeg]
exe_sim = [m for s,m in exes]
ocurrences_gen = {}
for x in exeg: 
    if x[1] not in ocurrences_gen.keys():
        ocurrences_gen[x[1]] = 1
    else:
        ocurrences_gen[x[1]] +=1
ocurrences_sim = {}
for x in exes: 
    if x[1] not in ocurrences_sim.keys():
        ocurrences_sim[x[1]] = 1
    else:
        ocurrences_sim[x[1]] +=1

sol_g = set(exe_gen)


prom_gen = sum(exe_gen)/len(exe_gen)
prom_sim = sum(exe_sim)/len(exe_sim)
dev_gen = 0
dev_sim = 0
for sol in exe_gen:
    dev_gen+=(abs(sol-prom_gen)**2)/len(exe_gen)
for sol in exe_sim:
    dev_sim+=(abs(sol-prom_sim)**2)/len(exe_sim)
dev_gen = math.sqrt(dev_gen)
dev_sim = math.sqrt(dev_sim)
print("Análisis de los Datos")
print("-----------------------------------")
print("Ocurrencias de las soluciones Algoritmo Genético")
print("m"," #(m)")
for s in sol_g:
    print(s,ocurrences_gen[s])
print("Probabilidad de Acierto:",ocurrences_gen[M]/n_exe)
print("Dispersion:",dev_gen)
print()
print("Ocurrencias de las soluciones Enfriamiento Simulado")
sol_s = set(exe_sim)
print("m"," #(m)")
for s in sol_s:
    print(s,ocurrences_sim[s])
print("Probabilidad de Acierto:",ocurrences_sim[M]/n_exe)
print("Dispersion:", dev_sim)