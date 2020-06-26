# folder = "20190419_1134"
folder = "20200625_16302"
  # os.makedirs("../../results/model/"+folder)
rho     = np.linspace(0,0.3,51 )  # link density in erdos renyi network
phi     = np.linspace(1,1.5,51)   # efficiency of coordinated Workers
# pe_range= np.logspace(np.log10(0.0001),np.log10(.02),21)
pe_explore = np.logspace(-4,-1.6,num = 51)
pe_null = np.array([0])

pe_range = np.concatenate((pe_null, pe_explore), axis=None)
pargrid = it.product(rho, phi)

print(pe_range)
input("press enter")
