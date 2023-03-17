#IMPORTS
import numpy as np
import matplotlib.pyplot as plt

#FUNCTIONS

def J(x,y,z,vx,vy,vz):
    r = [x,y,z]
    v = [vx,vy,vz]
    Jijk = np.cross(r,v)
    J = Jijk[0]+Jijk[1]+Jijk[2]
    return J


def semi_major_axis(E):
    a = -G*M/(2*E)
    return a

def eccentricity(E,J):
    e = (np.sqrt((G*M)**2 + (2*E*J**2)))/(G*M) 
    return e


def radius_eq(a,e,phi,phi0):
    r = (a*(1-e**2)) / (1+(e*np.cos(phi-phi0)))
    return r

def specific_E(vx,vy,vz,x,y,z):
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    rad = r(x,y,z)
    E = (0.5*v**2) - (G*M/rad)   
    return E


def a(p,x,y,z):
    radius = r(x,y,z)
    a =  - (G*M*p) / (radius**3)
    return a


def r(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return r


def euler_x(x,v,h):
    xprime = x+(v*h)
    return xprime


def euler_v(v,a,h):
    vprime = v+(a*h)
    return vprime


def torb(r):
    t = 2*np.pi*np.sqrt(r**3/(G*M))
    return t


def leapfrog(x,y,z,vx,vy,vz,dt):
    ax = a(x,x,y,z)
    ay = a(y,x,y,z)
    az = a(z,x,y,z)
    
    vx_prime_temp = vx + ax*dt/2
    vy_prime_temp = vy + ay*dt/2
    vz_prime_temp = vz + az*dt/2
    
    x_prime = x + (vx_prime_temp * dt)
    y_prime = y + (vy_prime_temp * dt)
    z_prime = z + (vz_prime_temp * dt)
    
    ax_prime = a(x_prime,x_prime,y_prime,z_prime)
    ay_prime = a(y_prime,x_prime,y_prime,z_prime)
    az_prime = a(z_prime,x_prime,y_prime,z_prime)
    
    vx_prime = vx_prime_temp + ax_prime*dt/2
    vy_prime = vy_prime_temp + ay_prime*dt/2
    vz_prime = vz_prime_temp + az_prime*dt/2
    
    return x_prime,y_prime,z_prime,vx_prime,vy_prime,vz_prime

#INITIAL CONDITIONS

G = 6.67e-11  
M = 2e30      

x0 = 1.5e11
y0 = 0
z0 = 0

vx0 = 0
vy0 = 15000
vz0 = 0

dt = 1000

Ang_mom = J(x0,y0,z0,vx0,vy0,vz0)
E_initial = specific_E(vx0,vy0,vz0,x0,y0,z0)

e = eccentricity(E_initial, Ang_mom)
sma = semi_major_axis(E_initial)
phi = np.linspace(0,2*np.pi,dt)

r_data = radius_eq(sma,e,phi,0)
x = r_data*np.cos(phi)
y = r_data*np.sin(phi)

plt.figure()
plt.plot(x,y)
plt.plot(0,0,'x')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()


steps = 1* int( torb(sma)/dt)

dt_array = [1e1,1e2,5e2,1e3,5e3,1e3]#1e4] #for actual runs

#dt_array = [1e1,5e1,1e2,5e2,1e3,5e3,1e4,5e4,1e5] #for error plots

error = []
j_error = []
error_sym = []
j_error_sym = []
error_leap = []
j_error_leap = []

for i in range(len(dt_array)):
    #Euler
    x_data_eul = np.zeros(steps)
    y_data_eul= np.zeros(steps)
    z_data_eul = np.zeros(steps)
    vx_data_eul = np.zeros(steps)
    vy_data_eul = np.zeros(steps)
    vz_data_eul = np.zeros(steps)
    
    x_data_eul[0] = x0
    y_data_eul[0] = y0
    z_data_eul[0] = z0
    vx_data_eul[0] = vx0
    vy_data_eul[0] = vy0
    vz_data_eul[0] = vz0
        
    #Symplectic
    x_data_sym = np.zeros(steps)
    y_data_sym = np.zeros(steps)
    z_data_sym = np.zeros(steps)
    vx_data_sym = np.zeros(steps)
    vy_data_sym = np.zeros(steps)
    vz_data_sym = np.zeros(steps)    
    
    x_data_sym[0] = x0
    y_data_sym[0] = y0
    z_data_sym[0] = z0
    vx_data_sym[0] = vx0
    vy_data_sym[0] = vy0
    vz_data_sym[0] = vz0
    
    #Leapfrog
    x_data_leap = np.zeros(steps)
    y_data_leap = np.zeros(steps)
    z_data_leap = np.zeros(steps)
    vx_data_leap = np.zeros(steps)
    vy_data_leap = np.zeros(steps)
    vz_data_leap = np.zeros(steps)    
    
    x_data_leap[0] = x0
    y_data_leap[0] = y0
    z_data_leap[0] = z0
    vx_data_leap[0] = vx0
    vy_data_leap[0] = vy0
    vz_data_leap[0] = vz0
     
    t = np.zeros(steps) 
    dt = dt_array[i]
    for i in range(steps-1):
        t[i+1] = t[i] + dt
        
        #Euler
        x_data_eul[i+1] = euler_x(x_data_eul[i],dt,vx_data_eul[i])
        y_data_eul[i+1] = euler_x(y_data_eul[i],dt,vy_data_eul[i])
        z_data_eul[i+1] = euler_x(z_data_eul[i],dt,vz_data_eul[i])
        
        vx_data_eul[i+1] = euler_v(vx_data_eul[i], dt, a(x_data_eul[i],x_data_eul[i],
                                                 y_data_eul[i],z_data_eul[i]))
        vy_data_eul[i+1] = euler_v(vy_data_eul[i], dt, a(y_data_eul[i],x_data_eul[i],
                                                 y_data_eul[i],z_data_eul[i]))
        vz_data_eul[i+1] = euler_v(vz_data_eul[i], dt, a(z_data_eul[i],x_data_eul[i],
                                                 y_data_eul[i],z_data_eul[i]))
                               
        
        #Symplectic
        vx_data_sym[i+1] = euler_v(vx_data_sym[i], dt, a(x_data_sym[i],
                                                            x_data_sym[i],
                                                            y_data_sym[i],
                                                            z_data_sym[i]))
        vy_data_sym[i+1] = euler_v(vy_data_sym[i], dt, a(y_data_sym[i],
                                                            x_data_sym[i],
                                                            y_data_sym[i],
                                                            z_data_sym[i]))
        vz_data_sym[i+1] = euler_v(vz_data_sym[i], dt, a(z_data_sym[i],
                                                            x_data_sym[i],
                                                            y_data_sym[i],
                                                            z_data_sym[i]))
        
            
        x_data_sym[i+1] = euler_x(x_data_sym[i],dt,vx_data_sym[i+1])
        y_data_sym[i+1] = euler_x(y_data_sym[i],dt,vy_data_sym[i+1])
        z_data_sym[i+1] = euler_x(z_data_sym[i],dt,vz_data_sym[i+1])
        
        
        #Leapfrog
        x_data_leap[i+1] = leapfrog(x_data_leap[i], y_data_leap[i], 
                                    z_data_leap[i], vx_data_leap[i],
                                    vy_data_leap[i], vz_data_leap[i], dt)[0]
        y_data_leap[i+1] = leapfrog(x_data_leap[i], y_data_leap[i], 
                                    z_data_leap[i], vx_data_leap[i],
                                    vy_data_leap[i], vz_data_leap[i], dt)[1]
        z_data_leap[i+1] = leapfrog(x_data_leap[i], y_data_leap[i], 
                                    z_data_leap[i], vx_data_leap[i],
                                    vy_data_leap[i], vz_data_leap[i], dt)[2]
        vx_data_leap[i+1] = leapfrog(x_data_leap[i], y_data_leap[i], 
                                    z_data_leap[i], vx_data_leap[i],
                                    vy_data_leap[i], vz_data_leap[i], dt)[3]
        vy_data_leap[i+1] = leapfrog(x_data_leap[i], y_data_leap[i], 
                                    z_data_leap[i], vx_data_leap[i],
                                    vy_data_leap[i], vz_data_leap[i], dt)[4]
        vz_data_leap[i+1] = leapfrog(x_data_leap[i], y_data_leap[i], 
                                    z_data_leap[i], vx_data_leap[i],
                                    vy_data_leap[i], vz_data_leap[i], dt)[5]

        
    E_final = specific_E(vx_data_eul[-1], vy_data_eul[-1], vz_data_eul[-1],\
                     x_data_eul[-1], y_data_eul[-1], z_data_eul[-1])
    relativee = abs((E_initial - E_final)/E_initial)* 100

    error.append(relativee)
    

    E_final_sym = specific_E(vx_data_sym[-1], vy_data_sym[-1], vz_data_sym[-1],\
                         x_data_sym[-1], y_data_sym[-1], z_data_sym[-1])
    relativee_sym = abs((E_initial - E_final_sym)/E_initial)* 100

    error_sym.append(relativee_sym)
    
    
    E_final_leap = specific_E(vx_data_leap[-1], vy_data_leap[-1], vz_data_leap[-1],\
                     x_data_leap[-1], y_data_leap[-1], z_data_leap[-1])
    relativee_leap = abs((E_initial - E_final_leap)/E_initial)* 100

    error_leap.append(relativee_leap)
    
    
    J_final = J(x_data_eul[-1], y_data_eul[-1], z_data_eul[-1],\
                vx_data_eul[-1], vy_data_eul[-1], vz_data_eul[-1]) 
    relativeJ = abs((Ang_mom - J_final)/Ang_mom)* 100
    
    j_error.append(relativeJ)
    

    J_final_sym = J(x_data_sym[-1], y_data_sym[-1], z_data_sym[-1],\
                vx_data_sym[-1], vy_data_sym[-1], vz_data_sym[-1]) 
    relativeJ_sym = abs((Ang_mom - J_final_sym)/Ang_mom)* 100
    
    j_error_sym.append(relativeJ_sym)
    
    
    J_final_leap = J(x_data_leap[-1], y_data_leap[-1], z_data_leap[-1],\
                vx_data_leap[-1], vy_data_leap[-1], vz_data_leap[-1]) 
    relativeJ_leap = abs((Ang_mom - J_final_leap)/Ang_mom)* 100
    
    j_error_leap.append(relativeJ_leap)
    

#FIGURES

plt.plot(x_data_eul,y_data_eul, label='Forward Euler')
plt.plot(x_data_sym,y_data_sym, label = 'Symplectic Euler')
plt.plot(x_data_leap,y_data_leap, label = 'Leapfrog')
plt.plot(0,0,'x')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.show()

plt.plot(x_data_sym,y_data_sym, label = 'Symplectic Euler', color="orange")
plt.plot(x_data_leap,y_data_leap, label = 'Leapfrog', color="green")
plt.plot(0,0,'x')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.show()

plt.plot(dt_array, error, label='Euler E', color='b')
plt.plot(dt_array, j_error, label='Euler J', color='b', ls=':')
plt.plot(dt_array, error_sym, label='Symplectic E', color='r')
plt.plot(dt_array, j_error_sym, label='Symplectic J', color='r', ls=':')
plt.plot(dt_array, error_leap, label='Leapfrog E', color='g')
plt.plot(dt_array, j_error_leap, label='Leapfrog J', color='g', ls=':')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Error (%)')
plt.xlabel('Time-step (s)')
plt.gca().invert_xaxis()
plt.show()




