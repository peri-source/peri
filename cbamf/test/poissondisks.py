import numpy as np

def float_to_index(float_tuple):
    return np.round(float_tuple).astype('int')


class DiskCollection:
    """
    Attributes:
    -----------
    self.min_sep:       float
        The min separation between disks.
    
    self.disk_grid:     numpy.ndarray
        The grid of the disk positions; 0 if there is no disk at the
        nearest pixel, 1 if it's at the nearest pixel.
    
    self.positions:      numpy.ndarray
        [N,3] array of the N disks' positions.
    
    Right now only works for 2 and 3D arrays; could be generalized.
    """
    def __init__(self, grid_shape, min_sep, max_sep=None, k=30):
        """
        Calculates a collection of hard disks that are not overlapping according
        to Bridson Siggraph 2007 "Fast Poisson Disk Sampling in Arbitrary Dimensions"

        Parameters:
        ----------
        grid_shape : tuple, array_like
            shape of the shape (in real coordinates) in which to space spheres

        min_sep : float
            minimum separation between objects

        max_sep : float
            maximum separation between objects

        k : int
            maximum number of rejections before moving on

        Returns:
        --------
        self : DiskCollection
        """
        if len(grid_shape) > 3:
            raise AttributeError("Can only handle 2D / 3D grids")

        #Step -1: Defininig some parameters:
        self.min_sep = min_sep
        if max_sep == None:
            self.max_sep = 2*self.min_sep
        else:
            self.max_sep = max_sep

        #Step 0. Initialize:
        self.disk_grid = np.zeros(np.array(grid_shape) + 2*min_sep, dtype ='int')

        #Step 1. Select an initial sample; initialize active list etc:
        start_pos = np.random.rand(len(grid_shape)) * np.array(grid_shape)
        start_ind = float_to_index(start_pos)
        self.disk_grid[tuple(start_ind)] = 1
        positions = [start_pos] ; active_list = [start_pos + min_sep]

        #Step 2: Loop over all elements in the active list:
        while len(active_list) > 0:
            #1. Pick a random point in the list and see if we can add a particle
            #   near it:
            np.random.shuffle( active_list )
            cur_pos = active_list.pop() 
            #2. Generate up to k random points uniformly from the spherical
            #   annulus on (minsep, maxsep); try to put them on:
            for attempt in xrange(k): 
                #Generate a normalized attempt of the correct magnitude
                this_point = np.random.randn(len(self.disk_grid.shape))
                this_point /= np.sqrt((this_point**2).sum())
                this_point *= min_sep + np.random.rand()*(self.max_sep - min_sep)
                this_point += cur_pos 

                overlapped = self.check_overlap(this_point)
                if not overlapped:
                    active_list.append( cur_pos )
                    active_list.append( this_point )

                    # un-pad the position
                    positions.append( this_point - min_sep )
                    self.disk_grid[tuple(float_to_index(this_point))] = len(positions) - 1
                    continue

        #And I need to un-pad...
        self.disk_grid = self.disk_grid[min_sep:-min_sep, min_sep:-min_sep].copy()
        self.positions = np.array(positions)

    def check_overlap( self, pos_to_check ):
        """
        Check for overlaps of a certain position
        """
        ind = float_to_index( pos_to_check )
        if np.any( ind <= self.min_sep ) or np.any( ind >= np.array( \
            self.disk_grid.shape) - self.min_sep ):
            return True
            #^checking if it's outside the image as an overlap
        pts = np.arange( -self.min_sep, self.min_sep +1)
        ms = self.min_sep #I don't want this to be infinity lines
        toReturn = False #No overlap as default

        # for pos_to_check in active_list:
        if len(ind) == 2:
            x,y = np.meshgrid( pts, pts )
            mask = x**2 + y**2 < ms**2
            cur_grid = self.disk_grid[ ind[0]-ms:ind[0]+ms+1,\
                ind[1]-ms:ind[1]+ms+1]
        elif len(ind) == 3:
            x,y,z = np.meshgrid( pts, pts, pts )
            mask = x**2 + y**2 + z**3 < ms**3
            cur_grid = self.disk_grid[ ind[0]-ms:ind[0]+ms+1,ind[1]-ms:ind[1]+\
                ms+1, ind[2] - ms:ind[2]+ms + 1]

        #checks using self.disk_grid
        toReturn = toReturn or np.any( cur_grid * mask )
        return toReturn

    def get_positions(self):
        return np.array(self.positions)


class PolydisperseDiskCollection:
    """
    Calculates a collection of _polydisperse_ hard disks that are not 
    overlapping. Uses a modified version of the algorith in Bridson Siggraph
    2007 "Fast Poisson Disk Sampling in Arbitrary Dimensions"
    
    Attributes:
    -----------
    self.disk_grid:     numpy.ndarray
        Binary grid of the disk positions; 0 if there is no disk at the 
        nearest pixel, 1 if there is one.
    self.positions:     numpy.ndarray
        N-element list of the N discs' positions.
    self.radii:         numpy.ndarray
        N-element list of the N discs' radii.

    Returns:
    --------
    self : DiskCollection
    
    Comments:
    ---------
    Right now this only works for 2 and 3D arrays but it could be generalized. 
    This is not a rigorous thing... if you attempt to plant disks forever 
    eventually there will always be a small disk at the tails of the 
    distribution that will fit into a crack. However this code will give
    something that is polydisperse and "mostly done."
    """
    def __init__( self, grid_shape, get_radii, pad, k=30, max_sep_factor=1.0):
        """
        Calculates a collection of _polydisperse_ hard disks that are not 
        overlapping. Uses a modified version of the algorith in Bridson Siggraph
        2007 "Fast Poisson Disk Sampling in Arbitrary Dimensions"

        Parameters:
        ----------
        grid_shape :    tuple, array_like
            shape of the shape (in real coordinates) in which to space spheres
        
        get_radii:      function
            get_radii() should return a radius for a random particle attempt. 
        
        pad:            int
            The maximum size of the particle. This is enforced internally.
        
        k:              int
            maximum number of rejections before moving on

        max_sep_factor : float
            The size of the step to take. Not sure if it does anything....
        """
        if len(grid_shape) > 3:
            raise AttributeError("Can only handle 2D / 3D grids")

        #Step 0. Initialize:
        self.pad = pad
        self.disk_grid = np.zeros(np.array(grid_shape) + 2*pad,dtype ='int')
        counter = 1
        
        #Step 1. Select an initial sample; initialize active list etc:
        start_pos = np.random.rand(len(grid_shape)) * np.array(grid_shape)
        start_ind = float_to_index( start_pos )
        self.disk_grid[tuple(start_ind)] = counter
        #The var active_list will be the padded positions, the var positions 
        #the raw positions. 
        active_list = [[start_pos + pad, 1*counter]]
        counter += 1
        positions = [start_pos]
        radii = [np.clip(get_radii(), 0, pad)]
        
        #Step 2: Loop over all elements in the active list:
        while len( active_list ) > 0:
            #1. Pick a random point in the list and see if we can add 
            #   a particle near it:
            la = len(active_list)
            this_ind = np.random.choice(range(la))
            cur_pos, part_lbl = active_list.pop(this_ind)

            #2. Generate up to k random points uniformly from the spherical 
            #   annulus on (minsep, 2*minsep); try to put them on:
            for attempt in xrange(k):
                #Generate a normalized attempt of the correct magnitude
                this_radius= np.clip(get_radii(), 0, pad)
                min_sep = 0.5 * (this_radius + radii[part_lbl-1])
                this_point = np.random.randn(len(self.disk_grid.shape))
                this_point /= np.sqrt((this_point**2).sum())#normalize
                this_point *= min_sep*(1.0 + max_sep_factor*np.random.rand() )
                this_point += cur_pos #the correct offset

                overlapped = self.check_overlap( this_point, min_sep)
                if not overlapped:
                    #First, replace the popped particle:
                    active_list.insert( this_ind, [cur_pos, part_lbl] )
                    active_list.append( [this_point, counter] )

                    #Add the radii, de-padded position to the cumulative list:
                    positions.append( this_point - pad )
                    radii.append( this_radius )
                    self.disk_grid[tuple(float_to_index(this_point))]=counter
                    counter += 1
                    continue

        #And I need to un-pad...
        self.disk_grid = self.disk_grid[min_sep:-min_sep, min_sep:-min_sep].copy()
        self.positions = np.array(positions)
        self.radii = np.array(radii)

    def check_overlap(self, pos_to_check, min_sep):
        ms = int(np.ceil(min_sep)) 
        ind = float_to_index(pos_to_check)
        
        #We call outside the image an overlap:
        if np.any(ind <= self.pad) or np.any(ind >= \
                np.array(self.disk_grid.shape) - self.pad ):
            return True

        pts = np.arange(-ms, ms+1)
        if len(ind) == 2:
            #2D
            x,y = np.meshgrid(pts, pts)
            mask = x**2 + y**2 < ms**2
            cur_grid = self.disk_grid[ind[0]-ms:ind[0]+ms+1,ind[1]-ms:ind[1]+ms+1]
        elif len(ind) == 3:
            #3D
            x,y,z = np.meshgrid( pts, pts, pts )
            mask = x**2 + y**2 + z**3 < ms**3
            cur_grid = self.disk_grid[ ind[0]-ms:ind[0]+ms+1,ind[1]-ms:ind[1]+\
                ms+1, ind[2] - ms:ind[2]+ms + 1]

        #FIXME this is too stringent -- it's just checking if it's within 1 px
        #of overlapping
        return np.any(cur_grid * mask) 
