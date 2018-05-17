import matplotlib.pyplot as plt
def velplotter(image,position1,position2,velocity,j):
    #fig1 = plt.figure()
    plt.imshow(image)
    plt.plot(position1[:,0],position1[:,1],"ro",markersize = 0.5)
    plt.quiver(position1[:,0],position1[:,1],velocity[:,0],velocity[:,1])
    #plt.show()
    #fig1.savefig('velstep-00'+str(startframe+j)+'0.pdf')                    
def squareplot(x,y,dgx,dgy):
    plt.plot([x,x+dgx,x+dgx,x,x],[y,y,y+dgy,y+dgy,y],"k-")
def plotensemble(grid,gx,gy,dgx,dgy):
    for ix in range(gx):
        for iy in range(gy):
            if grid[ix,iy,2] != 0:
                squareplot(grid[ix,iy,3],grid[ix,iy,4],dgx,dgy)
def plotter(nframes,grid,gridstuff,velpos = None, imgs = None):
    gx,gy,dgx,dgy = gridstuff[0],gridstuff[1],gridstuff[2],gridstuff[3]
  
    print("Results are ready, show plots? Yes will plot last frame and will include everything ")
    plot = input("(1) Yes, (2) No, (3) Advanced: ")
    F2P = nframes-2
    ens = 1
    if velpos == None:
        velplot = None
    else:
        velplot = 1
    cent = 1
    if plot == 3:
        print "Max allowed frame is",nframes-2,": "
        F2P = input("Choose frame: ")
        if velpos != None:
            velplot = input("Plot cells and velocity of cells? (1) Yes, (2) No: ")
        ens = input("Plot ensembles and velocity of ensemble? (1) Yes, (2) No: ")
        cent = input("Plot evolution of grid center? (1) Yes, (2) No: ")
      
    if plot != 2:
        #Grid centers plot
        if cent == 1:
            for k in range(nframes):
                for x in range(len(grid[k])): 
                    for y in range(len(grid[k])):
                        if grid[k,x,y,2] != 0:
                            plt.plot(grid[k,x,y,3]+dgx/2,grid[k,x,y,4]+dgy/2,"bo", markersize = 0.8)
                   
        if ens == 1:
           plotensemble(grid[F2P],gx,gy,dgx,dgy)
           for x in range(len(grid[F2P])): 
               for y in range(len(grid[F2P])):
                   if grid[F2P,x,y,2] != 0:
                       plt.quiver(grid[F2P,x,y,3]+dgx/2,grid[F2P,x,y,4]+dgy/2,grid[F2P,x,y,0],grid[F2P,x,y,1]) #the minus fixes axis problem
                       #plt.text((dgx*x)+gridfac/2,(dgy*y)+gridfac/2,str(grid[8,x,y,2].astype(int)),size = 7) #for plotting the number of counted cells in a gridcell
        if velplot == 1:
            velplotter(imgs[F2P],velpos[F2P,1],velpos[F2P,2],velpos[F2P,0],F2P)  
        plt.show()
