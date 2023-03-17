#Scipt that generates the initial conditions of n bodies using the Plummer 
#model and saves the output to a csv file called out.csv. Csv file can be 
#imported into Star_Cluster_Class.py to ensure multiple tests can be run
#using the same initial conditions by setting inputFile==True in 
#Star_Cluster_Class.py.
import numpy as np
import pandas as pd

def frand(low, high):
    return low + np.random.rand() * (high - low)

        
def mkplummer(n):
    initial_mass = []
    initial_posx = []
    initial_posy = []
    initial_posz = []
    initial_velx = []
    initial_vely = []
    initial_velz = []
    for i in range(n):
        initial_mass.append(1.0/n)
        radius = 1.0/np.sqrt(np.random.rand() ** (-2.0/3.0)-1)
        theta = np.arccos(frand(-1,1))
        phi = frand(0, 2*np.pi)
        scale_factor = 1
        initial_posx.append(((radius * np.sin(theta) * np.cos(phi))/scale_factor))
        initial_posy.append(((radius * np.sin(theta) * np.sin(phi))/scale_factor))
        initial_posz.append(((radius * np.cos(theta))/scale_factor))
        x = 0.0
        y = 0.1
        while y > x*x*(1.0-x*x)**3.5:
            x = frand(0,1)
            y = frand(0,0.1)
        velocity = x * np.sqrt(2.0) * (1.0 + radius*radius)**(-0.25)
        theta = np.arccos(frand(-1,1))
        phi = frand(0, 2*np.pi)
        initial_velx.append(((velocity * np.sin(theta) * np.cos(phi))/np.sqrt(scale_factor)))
        initial_vely.append(((velocity * np.sin(theta) * np.sin(phi))/np.sqrt(scale_factor)))
        initial_velz.append(((velocity * np.cos(theta))/np.sqrt(scale_factor)))
        
    return (initial_mass, initial_posx, initial_posy, initial_posz, initial_velx, initial_vely, initial_velz)

initial_mass, initial_posx, initial_posy, initial_posz, initial_velx, initial_vely, initial_velz = mkplummer(40)        
df = pd.DataFrame({
    'initial_mass':initial_mass,
    'initial_posx':initial_posx,
    'initial_posy':initial_posy,
    'initial_posz':initial_posz,
    'initial_velx':initial_velx,
    'initial_vely':initial_vely,
    'initial_velz':initial_velz
})
df.to_csv('out.csv', index=False)