#Script that performs convergence tests of total energy and total angular 
#momentum over a range of time-steps for the evolution of a single star 
#cluster. To change simulation variables, see bottom of script.
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import time 


class Body:
    def __init__(self, mass=0):
        '''Initialises the parameters of the body.'''
        self.mass = mass
        self.pos = [[],[],[]]
        self.vel = [[],[],[]]
        

class Nbody:
    default_seed = 123456
    
    h = [1e1, 1e0, 1e-1, 1e-2, 1e-3]
    
    G = 1
    G_unit = sc.G
    Msun = 1.98e30
    parsec = 3.086e16
    
    #mass unit in kg, cluster mass of 1e4Msun
    M_unit = 1e4*Msun
    print(M_unit, "M")
    #length unit in m, scale length of cluster of 1pc
    L_unit = 1*parsec
    print(L_unit, "L")
    #velocity unit in m/s, equal to root(GM/L)
    V_unit = np.sqrt(G_unit*M_unit/L_unit)
    print(V_unit, "V")
    #time unit in yrs, equal to root(L**3 / (MG))
    T_unit = np.sqrt(L_unit**3 / (M_unit*G_unit))/(365*24*60*60)
    print(T_unit, "T")

    softening = 1e16
    
    Total_energy_perc = []
    Total_ang_perc = []
    
    #bodies = []
    def __init__(self,ncluster,nstar,iterations):
        '''Initialisation function:
            ncluster - int - dictates the number of star clusters to be 
            modelled. Currently only 1 or 2 acceptable.
            nstar - int - dictates the number of stars that will be present
            in each cluster.
            iterations - int - dictates the number of iterations that will be
            simulated.
            importFile - boolean - dictates whether the function generates
            the initial conditions in the run (False) or imports the initial
            from the csv file "out.csv" generated from the Plummer_runner.py
            script (True).'''
        for hval in self.h:
            self.dt = hval*self.T_unit
            self.bodies = []
            self.total_star = ncluster*nstar
            self.hamiltonian = []
            self.potential = []
            self.kinetic = []
            self.ang_mom = []
            
            for i in range(self.total_star):
                self.bodies.append(Body())
                
            self.mkplummer(self.total_star,self.default_seed)       
            self.cluster_position_updater(nstar, ncluster)
            self.cluster_velocities_updater(nstar, ncluster)
            
            self.system_com()
            
            for i in range(iterations):
                if i % 5000 == 0:
                    print("Iteration number ", i, "reached.")
                self.Nbody_main(i, nstar, ncluster)
            
            self.Total_energy_perc.append(abs(((self.hamiltonian[-1]-self.hamiltonian[0])/self.hamiltonian[0])*100))
            self.Total_ang_perc.append(abs(((self.ang_mom[-1]-self.ang_mom[0])/self.ang_mom[0])*100))
        self.convergence_plotter()
        print("The run-time of the program is ", time.time() - start_time)
            
        
    def frand(self, low, high):
        return low + np.random.rand() * (high - low)
    
            
    def mkplummer(self, n, seed):
        '''Function to generate the initial conditions in phase space for n 
        bodies based off a Plummer distribution. Applied if importFile==False.'''
        if seed:
            self.seed = seed
        else:
            self.seed = self.default_seed
        
        for i in range(n):
            self.bodies[i].mass = (1.0/n)*self.M_unit
            radius = 1.0/np.sqrt(np.random.rand() ** (-2.0/3.0)-1)
            theta = np.arccos(self.frand(-1,1))
            phi = self.frand(0, 2*np.pi)
            scale_factor = 1
            self.bodies[i].pos[0].append(((radius * np.sin(theta) * np.cos(phi))/scale_factor)*self.L_unit)
            self.bodies[i].pos[1].append(((radius * np.sin(theta) * np.sin(phi))/scale_factor)*self.L_unit)
            self.bodies[i].pos[2].append(((radius * np.cos(theta))/scale_factor)*self.L_unit)
            x = 0.0
            y = 0.1
            while y > x*x*(1.0-x*x)**3.5:
                x = self.frand(0,1)
                y = self.frand(0,0.1)
            velocity = x * np.sqrt(2.0) * (1.0 + radius*radius)**(-0.25)
            theta = np.arccos(self.frand(-1,1))
            phi = self.frand(0, 2*np.pi)
            self.bodies[i].vel[0].append(((velocity * np.sin(theta) * np.cos(phi))/np.sqrt(scale_factor))*self.V_unit)
            self.bodies[i].vel[1].append(((velocity * np.sin(theta) * np.sin(phi))/np.sqrt(scale_factor))*self.V_unit)
            self.bodies[i].vel[2].append(((velocity * np.cos(theta))/np.sqrt(scale_factor))*self.V_unit)
            
            
    def acc_pot(self, xyzi, xyzj, mj):
        '''Returns the x,y and z acceleration on star i as a result of star j
        and the potential energy between the two stars.'''
        r = np.sqrt((xyzj[0]-xyzi[0])**2 + (xyzj[1]-xyzi[1])**2 + (xyzj[2]-xyzi[2])**2 + (self.softening)**2)
        ax = (self.G * mj * (xyzj[0]-xyzi[0]))/(r**3)
        ay = (self.G * mj * (xyzj[1]-xyzi[1]))/(r**3)
        az = (self.G * mj * (xyzj[2]-xyzi[2]))/(r**3)
        self.potential_pair = (-self.G * mj * mj)/((r**2+self.softening**2)**0.5)
        self.a = [ax,ay,az]
        
            
    def Nbody_main(self,k,nstar,ncluster):
        '''Function that determines the total energy, total potential energy
        and total angular momentum at each time-step. Initiates the update of 
        positions and velocities by running the symplectic_euler or leapfrog
        algorithms.'''
        #k is the iteration number   
        self.acc_dict = {}
        self.pot_dict = {}
        kinetic_storer = []
        ang_storer = []
        for i in range(len(self.bodies)):
            #print(i)
            self.acc_counterx = 0.
            self.acc_countery = 0.
            self.acc_counterz = 0.

            self.symplectic_euler(i, k)
            #self.leapfrog(i, k)
            
            kx = 0.5 * self.bodies[i].mass * (self.bodies[i].vel[0][k])**2
            ky = 0.5 * self.bodies[i].mass * (self.bodies[i].vel[1][k])**2
            kz = 0.5 * self.bodies[i].mass * (self.bodies[i].vel[2][k])**2
            kinetic_energy = np.sqrt((kx)**2 + (ky)**2 + (kz)**2)
            kinetic_storer.append(kinetic_energy)
            r = (self.bodies[i].pos[0][k],self.bodies[i].pos[1][k],self.bodies[i].pos[2][k])
            v = (self.bodies[i].vel[0][k],self.bodies[i].vel[1][k],self.bodies[i].vel[2][k])
            J = np.cross(r,v)
            ang_storer.append(J)
        kinetic_total = sum(kinetic_storer)
        ang_total = sum(ang_storer)
        potential_total = 0
        for value in self.pot_dict.values():
            potential_total+=value
         
        self.kinetic.append(kinetic_total)
        self.potential.append(potential_total)
        self.hamiltonian.append(kinetic_total+potential_total)
        self.ang_mom.append(sum(ang_total))

            
    def total_acceleration_calculator(self, i, k):
        '''Calculates the total acceleration acting on star i for a single
        iteration. '''
        #i is the initial star
        #j counts through the rest of the stars
        #k is the iteration number
        for j in range(len(self.bodies)):
            if i == j:
                pass
            elif (j,i) in self.acc_dict.keys():
                a_opp = []
                a_opp.append(-1*self.acc_dict[(j,i)][0])
                a_opp.append(-1*self.acc_dict[(j,i)][1])
                a_opp.append(-1*self.acc_dict[(j,i)][2])
                self.acc_dict[(i,j)] = a_opp
            else:
                xyzi = [self.bodies[i].pos[0][k],self.bodies[i].pos[1][k],self.bodies[i].pos[2][k]]
                xyzj = [self.bodies[j].pos[0][k],self.bodies[j].pos[1][k],self.bodies[j].pos[2][k]]
                self.acc_pot(xyzi,xyzj,self.bodies[i].mass)
                self.acc_dict[(i,j)] = self.a 
                self.pot_dict[(i,j)] = self.potential_pair

        for key,value in self.acc_dict.items():
            if key[0] == i:
                self.acc_counterx = self.acc_counterx + value[0]
                self.acc_countery = self.acc_countery + value[1]
                self.acc_counterz = self.acc_counterz + value[2]
            else:
                pass
        
    def symplectic_euler(self, i, k):
        '''Function that updates the positions and velocities of star i using
        the symplectic Euler method.'''
        #i is star
        #k is iteration value
        self.total_acceleration_calculator(i, k)
        self.bodies[i].vel[0].append(self.bodies[i].vel[0][k] + self.dt * self.acc_counterx)
        self.bodies[i].vel[1].append(self.bodies[i].vel[1][k] + self.dt * self.acc_countery)
        self.bodies[i].vel[2].append(self.bodies[i].vel[2][k] + self.dt * self.acc_counterz)
        self.bodies[i].pos[0].append(self.bodies[i].pos[0][k] + self.dt * self.bodies[i].vel[0][k+1])
        self.bodies[i].pos[1].append(self.bodies[i].pos[1][k] + self.dt * self.bodies[i].vel[1][k+1])
        self.bodies[i].pos[2].append(self.bodies[i].pos[2][k] + self.dt * self.bodies[i].vel[2][k+1])
        
        
    def leapfrog(self, i, k):
        '''Function that updates the positions and velocities of star i using
        the leapfrog algorithm.'''
        self.total_acceleration_calculator(i, k)
        
        vx_prime_temp = self.bodies[i].vel[0][k] + ((self.dt/2) * self.acc_counterx)
        vy_prime_temp = self.bodies[i].vel[1][k] + ((self.dt/2) * self.acc_countery)
        vz_prime_temp = self.bodies[i].vel[2][k] + ((self.dt/2) * self.acc_counterz)
        
        self.bodies[i].pos[0].append(self.bodies[i].pos[0][k] + (self.dt * vx_prime_temp))
        self.bodies[i].pos[1].append(self.bodies[i].pos[1][k] + (self.dt * vy_prime_temp))
        self.bodies[i].pos[2].append(self.bodies[i].pos[2][k] + (self.dt * vz_prime_temp))
        
        self.total_acceleration_calculator(i, k)
        
        self.bodies[i].vel[0].append(vx_prime_temp + ((self.dt/2) * self.acc_counterx))
        self.bodies[i].vel[1].append(vy_prime_temp + ((self.dt/2) * self.acc_countery))
        self.bodies[i].vel[2].append(vz_prime_temp + ((self.dt/2) * self.acc_counterz))
    
            
    def cluster_position_updater(self, nstar, ncluster):
        '''Function that updates the relative positions of the two star 
        clusters.'''
        self.locations = [[],[],[]]
        if ncluster == 1:
            self.locations[0].append(0)
            self.locations[1].append(0)
            self.locations[2].append(0)
            self.locations[0].append(0)
            self.locations[1].append(0)
            self.locations[2].append(0)
        else:
            self.locations[0].append(-1e17)
            self.locations[1].append(-1e17)
            self.locations[2].append(-1e17)
            self.locations[0].append(1e17)
            self.locations[1].append(1e17)
            self.locations[2].append(1e17)
            
        for j in range(self.total_star):
            if j < nstar:
                self.bodies[j].pos[0][0] = self.bodies[j].pos[0][0] + self.locations[0][0]
                self.bodies[j].pos[1][0] = self.bodies[j].pos[1][0] + self.locations[1][0]
                self.bodies[j].pos[2][0] = self.bodies[j].pos[2][0] + self.locations[2][0] 
            else:
                self.bodies[j].pos[0][0] = self.bodies[j].pos[0][0] + self.locations[0][1]
                self.bodies[j].pos[1][0] = self.bodies[j].pos[1][0] + self.locations[1][1]
                self.bodies[j].pos[2][0] = self.bodies[j].pos[2][0] + self.locations[2][1]
                
        
    def cluster_velocities_updater(self, nstar, ncluster):
        '''Function that updates the relative velocities of the two star 
        clusters.'''
        self.velocities = [[],[],[]]
        if ncluster == 1:
            self.velocities[0].append(0)
            self.velocities[1].append(0)
            self.velocities[2].append(0)
            self.velocities[0].append(0)
            self.velocities[1].append(0)
            self.velocities[2].append(0)
        else:
            self.velocities[0].append(6e7)
            self.velocities[1].append(6e7)
            self.velocities[2].append(6e7)
            self.velocities[0].append(-6e7)
            self.velocities[1].append(-6e7)
            self.velocities[2].append(-6e7)
            
        for j in range(self.total_star):
            if j < nstar:
                self.bodies[j].vel[0][0] = self.bodies[j].vel[0][0] + self.velocities[0][0]
                self.bodies[j].vel[1][0] = self.bodies[j].vel[1][0] + self.velocities[1][0]
                self.bodies[j].vel[2][0] = self.bodies[j].vel[2][0] + self.velocities[2][0]
            else:
                self.bodies[j].vel[0][0] = self.bodies[j].vel[0][0] + self.velocities[0][1]
                self.bodies[j].vel[1][0] = self.bodies[j].vel[1][0] + self.velocities[1][1]
                self.bodies[j].vel[2][0] = self.bodies[j].vel[2][0] + self.velocities[2][1]
                
                
                
    def system_com(self):
        '''Function that updates the positions and velocities of the two
        clusters so that the center of mass and center of momentum of the 
        system are at (0,0,0).'''
        self.com_calculator(0, 0, self.total_star)
        for i in range(self.total_star):
            self.bodies[i].pos[0][0] = self.bodies[i].pos[0][0] - self.rx
            self.bodies[i].pos[1][0] = self.bodies[i].pos[1][0] - self.ry
            self.bodies[i].pos[2][0] = self.bodies[i].pos[2][0] - self.rz
            self.bodies[i].vel[0][0] = self.bodies[i].vel[0][0] - self.vx
            self.bodies[i].vel[1][0] = self.bodies[i].vel[1][0] - self.vy
            self.bodies[i].vel[2][0] = self.bodies[i].vel[2][0] - self.vz
            
        
    def com_calculator(self, k, istar, nstar):
        '''Function that calculates the current center of mass and center of 
        momentum of the system.'''
        rx_t, ry_t, rz_t = 0, 0, 0
        vx_t, vy_t, vz_t = 0, 0, 0        
        for i in range(istar, nstar):
            rx_t = rx_t + (self.bodies[i].pos[0][k])
            ry_t = ry_t + (self.bodies[i].pos[1][k])
            rz_t = rz_t + (self.bodies[i].pos[2][k])  
            vx_t = vx_t + (self.bodies[i].vel[0][k])
            vy_t = vy_t + (self.bodies[i].vel[1][k])
            vz_t = vz_t + (self.bodies[i].vel[2][k]) 
        self.rx = rx_t / nstar
        self.ry = ry_t / nstar
        self.rz = rz_t / nstar
        self.vx = vx_t / nstar
        self.vy = vy_t / nstar
        self.vz = vz_t / nstar
            
        return [self.rx, self.ry, self.rz, self.vx, self.vy, self.vz]

        
    def convergence_plotter(self):
        '''Function that plots the total energy error and total angular
        momentum error for the evolution of a single star cluster for a 
        range of timesteps.'''
        plt.plot(self.h, self.Total_energy_perc, label='Total Energy Error')
        plt.plot(self.h, self.Total_ang_perc, label='Total Angular Momentum Error')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Time-step (s)")
        plt.ylabel("Error (%)")
        plt.gca().invert_xaxis()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        plt.legend()
        plt.show
        print(self.Total_energy_perc)
        print(self.Total_ang_perc)
            

if __name__ == "__main__":
    #Currently set to run 1 cluster with 20 stars for 5000 iterations. This 
    #module currently only works for 1 star cluster!!
    start_time = time.time()
    Nbody(1,20,5000)