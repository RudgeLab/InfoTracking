import numpy as np

class CellState:
    pass

class EnsembleState:
    def __init__(self, pos_x, pos_y, w, h):
        # Initialise the ensemble at a given location
        self.w = w
        self.h = h
        self.pos_x = pos_x
        self.pos_y = pos_y

class ImageEnsembleState(Ensemble):
    def __init__(self, pos_x, pos_y, w, h, image_roi):
        # Ensemble computed from image, stores region of interest at each t
        super(EnsembleState, self).__init__(pos_x, pos_y, w, h)
        self.image_roi = np.array((nt,w,h))
        self.image_roi[0,:,:] = image_roi
        self.cond_entropy

class ModelEnsembleState(Ensemble):
    def __init__(self, pos_x, pos_y, w, h, scale):
        # Ensemble from dictionary of CellModeller CellState objects
        super(EnsembleState, self).__init__(pos_x, pos_y, w, h)
        self.scale = scale # Scaling from model position to image coordinate
        self.cells = {}

    def add_cell(self, cell):
        self.cells[cell.id] = cell

    def compute_mean_velocity(self, cells2):
        # cells2 = dictionary of cell states in next time step
        ncells_common = 0
        vel_sum = np.zeros((2,))
        for (id,c) in self.cells:
            if id in cells2:
                ncells_common += 1
                vel_sum += cells2[id].pos - c.pos
            else:
                # Here we check parents of cells that have divided
                for (nid,cnew) in cells2:
                    if cnew.parent == id:
                        ncells_common += 1
                        vel_sum += cnew.pos - c.pos
        return vel_sum/ncells_common
                        

class EnsembleGrid:
    def __init__(self, gx0, gy0, w, h, scale, nt):
        self.grid_states = {}
        self.gx0 = gx0
        self.gy0 = gy0
        self.nx = nx
        self.ny = ny
        self.w = w
        self.h = h
        self.scale = scale
        self.nt = nt

class ModelEnsembleGrid(EnsembleGrid):
    def __init__(self, gx0, nx, ny, w, h, scale, pickle_fname, step, nt):
        super(EnsembleGrid, self)(gx0, gy0, w, h, nx, ny, scale, nt)
        # Initialise with given cells at time t=0
        fname = pickle_fname%(0)
        self.add_ensemble_states(fname%0, 0)
        for i in range(1,nt):
            t = i*step
            for ix in range(self.nx):
                for iy in range(self.ny):
                    es0 = self.grid_states[(gx,gy,t-step)]
                    cells = self.get_cell_states(fname%t)
                    es1 = ModelEnsembleState(
                    self.grid_states[(gx,gy,t-step)]
            vel = self.grid_states[

    def get_cell_states(self, picklefile):
        data = cPickle.load(fname)
        cells = data['cellStates']
        return cells
        
    def add_ensemble_states(self, fname, t)
        cells = self.get_cell_states(fname)
        # Construct the grid by looping over ensemble regions
        for ix in range(self.nx):
            for iy in range(self.ny)
                # Compute ensemble top-left position
                gx,gy = self.gx0 + self.w*ix, self.gy0 + self.h*iy
                # Create a new ensemble 
                es = ModelEnsembleState(gx,gy,self.w,self.h,self.scale)
                # Find and add all cells to the ensemble that are inside the boundary
                for (cid,cell) in self.cells:
                    spos = cell.pos * scale
                    if spos[0] >= gx and spod[1] >= gy and x < spos[0]+w and spos[1] < gy+h:
                        es.add_cell(cell)a
                # Store ensemble in dictionary indexed by grid location and time
                self.grid_states[(gx,gy,t)] = es

    def compute_mean_velocities(self, cells):
        for ((gx,gy,t),state) in self.grid_states.iteritems():
            vel = state.compute_mean_velocity(cells)
            gx2, gy2 = gx+vel[0], gy+vel[1]
            es = ModelEnsembleState(gx2,gy2,self.w,self.h,self.scale)



