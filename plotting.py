import matplotlib.pyplot as plt
def squareplot(x,y,dgx,dgy,i):
    plt.plot([x,x+dgx,x+dgx,x,x],[y,y,y+dgy,y+dgy,y],"k-",linewidth = 0.3)
    
def plotensemble(grid,t,gx,gy,dgx,dgy):
    for ix in range(gx):
        for iy in range(gy):
            if grid[t,ix,iy].actualcells != 0:
                squareplot(grid[t,ix,iy].px,grid[t,ix,iy].py,dgx,dgy,t)
                
def plotanimate(grid,imgs,gx,gy,dgx,dgy,p):
    for i in range(len(grid.frames)-1):
        plt.clf()
        plt.hold(True)
        plt.imshow(imgs[i],origin = 'lower')
        plotensemble(grid,i,gx,gy,dgx,dgy)
        plt.hold(False)
        plt.pause(p)
        
def plotter(grid,imgs):
    gx,gy,dgx,dgy = grid.gx,grid.gy,grid.dgx,grid.dgy
    plt.hold(True)
    print("Results are ready, show plots? Yes will plot last frame and will include everything ")
    plot = input("(1) Yes, (2) No, (3) Advanced, (4) Animate: ")
    F2P = grid.nframes-2
    ens = 1
    cent = 1
    velplot = 0
    cellplot = 1
    fig = plt.figure(figsize=(16,16))
    
    if plot == 4:
        pause = input("Type delay between plots: ")
        plotanimate(grid,imgs,gx,gy,dgx,dgy,pause)
    if plot == 3:
        print "Max allowed frame is",grid.nframes-2,": "
        F2P = input("Choose frame: ")
        ens = input("Plot ensembles and velocity of ensemble? (1) Yes, (2) No: ")
        cent = input("Plot evolution of grid center? (1) Yes, (2) No: ")
        cellplot = input("Plot cells? (1) Yes, (2) No: ")
        velplot = input("Plot velocity of cells? (1) Yes, (2) No: ")
    if plot != 2 and plot != 4:
        #Grid centers plot
        if cent == 1:
            for k in range(grid.nframes):
                for x in range(grid.gx): 
                    for y in range(grid.gy):
                        if grid[k,x,y].actualcells != 0:
                            plt.plot(grid[k,x,y].px + (grid.dgx)/2,grid[k,x,y].py+(grid.dgy)/2,"bo", markersize = 0.8)
                   
        if ens == 1:
           plotensemble(grid,F2P,gx,gy,dgx,dgy)
           for x in range(grid.gx): 
               for y in range(grid.gy):
                   if grid[F2P,x,y].actualcells != 0:
                       plt.quiver(grid[F2P,x,y].px+grid.dgx/2,grid[F2P,x,y].py+grid.dgy/2,grid[F2P,x,y].vx ,grid[F2P,x,y].vy) #the minus fixes axis problem
                       #plt.text((dgx*x)+gridfac/2,(dgy*y)+gridfac/2,str(grid[8,x,y,2].astype(int)),size = 7) #for plotting the number of counted cells in a gridcell
        if cellplot == 1:
            plt.imshow(imgs[F2P],origin = 'lower')
            plt.colormap("gray")
        if velplot == 1:
                for ix in range(grid.gx):
                    for iy in range(grid.gy):
                        for cell in grid[F2P,ix,iy].cells:
                            plt.plot(grid.resize*cell.pos[0]+grid.center,-grid.resize*cell.pos[1]+grid.center,"ro",markersize = 0.2)
                            #plt.quiver(cell.pos[0]+grid.center,cell.pos[1]+grid.center,cell.vx,cell.vy) #the minus fixes axis problem

    plt.show()
