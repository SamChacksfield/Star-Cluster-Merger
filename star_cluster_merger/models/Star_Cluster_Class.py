#Script to simulate the evolution of a merger of two star clusters. Outputs an
#animation of the merger and the density profile every 1000 iterations. After
#the simulation has finished, outputs the total energy, total potential energy,
#total kinetic energy and total angular momentum of the system. To change
#simulation variables, see bottom of script.
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import time 
import pandas as pd


class Body:
    def __init__(self, mass=0):
        '''Initialises the parameters of the body.'''
        self.mass = mass
        self.pos = [[],[],[]]
        self.vel = [[],[],[]]
        

class Nbody:
    default_seed = 123456
    h = 1 #timestep 
    
    G = 1
    G_unit = sc.G
    Msun = 1.98e30
    parsec = 3.086e16
    
    #mass unit in kg, cluster mass of 1e4Msun
    M_unit = 5e2*Msun
    print(M_unit, "M")
    #length unit in m, scale length of cluster of 1pc
    L_unit = 0.01*parsec
    print(L_unit, "L")
    #velocity unit in m/s, equal to root(GM/L)
    V_unit = np.sqrt(G_unit*M_unit/L_unit)
    print(V_unit, "V")
    #time unit in yrs, equal to root(L**3 / (MG))
    T_unit = np.sqrt(L_unit**3 / (M_unit*G_unit))/(365*24*60*60)
    print(T_unit, "T")

    dt = h*T_unit
    print(dt, "dt")
    softening = 1e14
    
    #bodies = []
    def __init__(self,ncluster,nstar,iterations, importFile=False):
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
        print("Start!!")
        self.bodies = []
        self.total_star = ncluster*nstar
        for i in range(self.total_star):
            self.bodies.append(Body())
         
        if importFile:
            self.apply_initial_condition()  
        else:
            self.mkplummer(self.total_star, self.default_seed)

        self.hamiltonian = []
        self.potential = []
        self.kinetic = []
        self.ang_mom = []
        
                  
        self.cluster_position_updater(nstar, ncluster)
        self.cluster_velocities_updater(nstar, ncluster)
        
        self.system_com()
        
        for i in range(iterations):
            if i % 5000 == 0:
                print("Iteration number ", i, "reached.")
            self.Nbody_main(i, nstar, ncluster)
            # if i % 1000 == 0:
            #     self.spherical_density_func(i)
            
            

        for i in range(len(self.bodies[0].pos[0])):
              if i % 1000 == 0:
                  self.visualisation(i, nstar, ncluster)   
            
        self.post_processing(iterations)
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
            
            
    def apply_initial_condition(self):
        '''Function that reads in a csv file of initial conditions generated
        using a Plummer distribution. Csv file is made in the script 
        Plummer_runner.py. Applied if importFile==True.'''
        values = pd.read_csv('out.csv')
        # values['initial_mass']
        for i in range(self.total_star):
            self.bodies[i].mass = values['initial_mass'][i]*self.M_unit
            self.bodies[i].pos[0].append(values['initial_posx'][i]*self.L_unit)
            self.bodies[i].pos[1].append(values['initial_posy'][i]*self.L_unit)
            self.bodies[i].pos[2].append(values['initial_posz'][i]*self.L_unit)
            self.bodies[i].vel[0].append(values['initial_velx'][i]*self.V_unit)
            self.bodies[i].vel[1].append(values['initial_vely'][i]*self.V_unit)
            self.bodies[i].vel[2].append(values['initial_velz'][i]*self.V_unit)

            
            
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
            self.locations[0].append(-1e15)
            self.locations[1].append(-1e15)
            self.locations[2].append(-1e15)
            self.locations[0].append(1e15)
            self.locations[1].append(1e15)
            self.locations[2].append(1e15)
            
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
            self.velocities[0].append(6e4)
            self.velocities[1].append(6e4)
            self.velocities[2].append(6e4)
            self.velocities[0].append(-6e4)
            self.velocities[1].append(-6e4)
            self.velocities[2].append(-6e4)
            
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
            
        ##com checker
        #self.com_calculator(0,0,self.total_star)
        #print(self.rx, self.ry, self.rz)
        
        
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
    

    def spherical_density_func(self, k):
        '''Function that calculates the density of stars within segments of 
        equal radius away from the origin. Plots the density against radius 
        from center every 1000 iterations.'''
        bin1 = 0
        bin2 = 0
        bin3 = 0
        bin4 = 0
        bin5 = 0
        bin6 = 0
        bin7 = 0
        bin8 = 0
        bin9 = 0
        bin10 = 0
        bin11 = 0
        bin12 = 0
        bin13 = 0
        bin14 = 0
        bin15 = 0
        bin16 = 0
        bin17 = 0
        bin18 = 0
        bin19 = 0
        bin20 = 0
        for i in range(self.total_star):
            r = np.sqrt((self.bodies[i].pos[0][k]**2) + (self.bodies[i].pos[1][k]**2) + (self.bodies[i].pos[2][k]**2))
            if r <= self.L_unit*8.0 and r > self.L_unit*3.80:
                bin1+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*6.0)**3) - (((4/3)*np.pi*(self.L_unit*3.80)**3))))
            elif r <= self.L_unit*3.80 and r > self.L_unit*3.60:
                bin2+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*3.8)**3) - (((4/3)*np.pi*(self.L_unit*3.60)**3))))
            elif r <= self.L_unit*3.60 and r > self.L_unit*3.40:
                bin3+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*3.6)**3) - (((4/3)*np.pi*(self.L_unit*3.40)**3))))
            elif r <= self.L_unit*3.40 and r > self.L_unit*3.20:
                bin4+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*3.4)**3) - (((4/3)*np.pi*(self.L_unit*3.20)**3))))
            elif r <= self.L_unit*3.20 and r > self.L_unit*3.0:
                bin5+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*3.2)**3) - (((4/3)*np.pi*(self.L_unit*3.0)**3))))
            elif r <= self.L_unit*3.0 and r > self.L_unit*2.80:
                bin6+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*3.0)**3) - (((4/3)*np.pi*(self.L_unit*2.80)**3))))
            elif r <= self.L_unit*2.80 and r > self.L_unit*2.60:
                bin7+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*2.8)**3) - (((4/3)*np.pi*(self.L_unit*2.60)**3))))
            elif r <= self.L_unit*2.60 and r > self.L_unit*2.40:
                bin8+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*2.6)**3) - (((4/3)*np.pi*(self.L_unit*2.40)**3))))
            elif r <= self.L_unit*2.40 and r > self.L_unit*2.20:
                bin9+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*2.4)**3) - (((4/3)*np.pi*(self.L_unit*2.20)**3))))
            elif r <= self.L_unit*2.20 and r > self.L_unit*2.0:
                bin10+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*2.2)**3) - (((4/3)*np.pi*(self.L_unit*2.0)**3))))
            elif r <= self.L_unit*2.0 and r > self.L_unit*1.80:
                bin11+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*2.0)**3) - (((4/3)*np.pi*(self.L_unit*1.80)**3))))
            elif r <= self.L_unit*1.80 and r > self.L_unit*1.60:
                bin12+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*1.8)**3) - (((4/3)*np.pi*(self.L_unit*1.60)**3))))
            elif r <= self.L_unit*1.60 and r > self.L_unit*1.40:
                bin13+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*1.6)**3) - (((4/3)*np.pi*(self.L_unit*1.40)**3))))
            elif r <= self.L_unit*1.40 and r > self.L_unit*1.20:
                bin14+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*1.4)**3) - (((4/3)*np.pi*(self.L_unit*1.20)**3))))
            elif r <= self.L_unit*1.20 and r > self.L_unit*1.0:
                bin15+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*1.2)**3) - (((4/3)*np.pi*(self.L_unit*1.0)**3))))
            elif r <= self.L_unit*1.0 and r > self.L_unit*0.80:
                bin16+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*1.0)**3) - (((4/3)*np.pi*(self.L_unit*0.80)**3))))
            elif r <= self.L_unit*0.80 and r > self.L_unit*0.60:
                bin17+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*0.8)**3) - (((4/3)*np.pi*(self.L_unit*0.60)**3))))
            elif r <= self.L_unit*0.60 and r > self.L_unit*0.40:
                bin18+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*0.6)**3) - (((4/3)*np.pi*(self.L_unit*0.40)**3))))
            elif r <= self.L_unit*0.40 and r > self.L_unit*0.20:
                bin19+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*0.4)**3) - (((4/3)*np.pi*(self.L_unit*0.20)**3))))
            elif r <= self.L_unit*0.20 and r > self.L_unit*0:
                bin20+=1*(self.bodies[i].mass / (((4/3)*np.pi*(self.L_unit*0.2)**3) - (((4/3)*np.pi*(self.L_unit*0)**3))))
            
        radial_dist = ['3.8','3.6','3.4','3.2','3.0', 
                       '2.8','2.6','2.4','2.2','2.0',
                       '1.8','1.6','1.4','1.2','1.0',
                       '0.8','0.6','0.4','0.2','0.0']
        star_num = [bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9,bin10,
                    bin11,bin12,bin13,bin14,bin15,bin16,bin17,bin18,bin19,bin20]
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(radial_dist,star_num)
        plt.xlabel("Radial Distance (1e^-2 pc)")
        plt.ylabel("Density (kg/m^3)")
        plt.show()
        print("Time histo ", self.dt * k, "plot number ", k)


    
    def visualisation(self,j,nstar,ncluster):
        '''Function that plots the positions of all bodies every 1000 
        iterations.'''
        ax = plt.axes(xlim=(-3e15,3e15),ylim=(-3e15,3e15),zlim=(-3e15,3e15),projection="3d")
        if ncluster == 1:
            for i in range(len(self.bodies)):
                ax.scatter(self.bodies[i].pos[0][j], 
                               self.bodies[i].pos[1][j], 
                               self.bodies[i].pos[2][j], s=5, c='r')
        else:
            for i in range(len(self.bodies)):
                if i<nstar:
                    ax.scatter(self.bodies[i].pos[0][j], self.bodies[i].pos[1][j], 
                               self.bodies[i].pos[2][j], s=5, c='r')
                else:
                    ax.scatter(self.bodies[i].pos[0][j], self.bodies[i].pos[1][j], 
                               self.bodies[i].pos[2][j], s=5, c='b')               
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        plt.show()
        #print("Plot number ", j)
        
    
    def post_processing(self, iterations):
        '''Function that handles the post processing of data, including the
        calculation of the following properties variation with time:
            Hamiltonian
            Potential energy
            Kinetic energy
            Total angular momentum
        '''
        time_list = []
        for i in range(iterations):
            time = i*self.dt
            time_list.append(time)
           
        fig, axs = plt.subplots(2,2)
        axs[0,0].plot(time_list, self.hamiltonian)
        axs[0,0].set_ylabel("Hamiltonian of Star Cluster (J)")
        axs[0,1].plot(time_list, self.potential)
        axs[0,1].set_ylabel("Potential of Star Cluster (J)")
        axs[1,0].plot(time_list, self.kinetic)
        axs[1,0].set_ylabel("Kinetic energy of Star Cluster (J)")
        axs[1,1].plot(time_list, self.ang_mom)
        axs[1,1].set_ylabel("Angular momentum of Star Cluster (m^2/s)")
        for ax in axs.flat:
            ax.set_xlabel("Time (years)")
        
        
        print(self.hamiltonian[0], "start E")
        print(self.hamiltonian[-1], "end E")
        print(min(self.hamiltonian), "min E")
        print(max(self.hamiltonian), "max E")
        print("Percentage difference E ", ((self.hamiltonian[-1]-self.hamiltonian[0])/self.hamiltonian[0])*100)
        print("Min Max perc dif E ", ((max(self.hamiltonian) - min(self.hamiltonian))/min(self.hamiltonian))*100)
        print(self.ang_mom[0], "start J")
        print(self.ang_mom[-1], "end J")
        print(min(self.ang_mom), "min J")
        print(max(self.ang_mom), "max J")
        print("Percentage difference J ", ((self.ang_mom[-1]-self.ang_mom[0])/self.ang_mom[0])*100)
        print("Min Max perc dif J ", ((max(self.ang_mom) - min(self.ang_mom))/min(self.ang_mom))*100)


if __name__ == "__main__":
    start_time = time.time()
    #Currently set to run 2 star clusters, both with 20 stars over a duration
    #of 500,000 iterations. Approximate run-time 2-3 hours. For faster runs, 
    #reduce the number of iterations and increase the value of the time-step
    #h. h=1, iterations=50,000 can be used to model quicker less accurate mergers
    nbody = Nbody(2,10,50000)