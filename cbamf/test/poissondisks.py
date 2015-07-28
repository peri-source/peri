import numpy as np

class DiskCollection:
    """
    self.min_sep: The min separation between disks.
    self.diskGrid: The grid of the disk positions; 0 if there is no disk at the
        nearest pixel, 1 if it's at the nearest pixel.
    self.positions: N-element list, each element in the list is an array of the
        corresponding disk's position.
    Right now only works for 2 and 3D arrays; could be generalized.
    """
    def __init__(self, gridshape, min_sep, max_sep=None, k=30):
        """
        Calculates a collection of hard disks that are not overlapping according
        to Bridson Siggraph 2007 "Fast Poisson Disk Sampling in Arbitrary Dimensions"

        Parameters:
        ----------
        gridshape : tuple, array_like
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
        if len(gridshape) > 3:
            raise AttributeError("Can only handle 2D / 3D grids")

        #Step -1: Defininig some parameters:
        self.min_sep = min_sep
        if max_sep == None:
            self.max_sep = 2*self.min_sep
        else:
            self.max_sep = max_sep

        #Step 0. Initialize:
        self.diskGrid = np.zeros( np.array( gridshape) + 2*min_sep, dtype ='int')

        #Step 1. Select an initial sample; initialize active list etc:
        startPos = np.random.rand(len(gridshape)) * np.array(gridshape)
        startInd = self.float_to_index( startPos )
        self.diskGrid[tuple(startInd)] = 1
        self.positions = [ startPos ] ; activeList = [ startPos +min_sep ]

        #Step 2: Loop over all elements in the active list:
        while len(activeList) > 0:
            #1. Pick a random point in the list and see if we can add a particle
            #   near it:
            np.random.shuffle( activeList )
            curPos = activeList.pop() #Popping it out; will put it back later
            #2. Generate up to k random points uniformly from the spherical
            #   annulus on (minsep, maxsep); try to put them on:
            for a in xrange(k): #with a continue statement if it works
                thisPoint = np.random.randn( len( self.diskGrid.shape ) )
                thisPoint /= np.sqrt( (thisPoint**2).sum() )#normalize
                thisPoint *= min_sep + np.random.rand()*(self.max_sep - min_sep)
                #^setting to correct magnitude
                thisPoint += curPos #the correct offset
                overlapped = self.check_overlap( thisPoint )
                #^also checks if it's in the image

                # if not overlapping, add the particle
                if not overlapped:
                    activeList.append( curPos )
                    activeList.append( thisPoint )

                    # un-pad the position
                    self.positions.append( thisPoint - min_sep )
                    self.diskGrid[tuple(self.float_to_index(thisPoint))] = len(self.positions) - 1
                    continue

        #And I need to un-pad...
        self.diskGrid = self.diskGrid[min_sep:-min_sep, min_sep:-min_sep]

    def check_overlap( self, posToCheck ):
        """
        Check for overlaps of a certain position
        """
        ind = self.float_to_index( posToCheck )
        if np.any( ind <= self.min_sep ) or np.any( ind >= np.array( \
            self.diskGrid.shape) - self.min_sep ):
            return True
            #^checking if it's outside the image as an overlap
        pts = np.arange( -self.min_sep, self.min_sep +1)
        ms = self.min_sep #I don't want this to be infinity lines
        toReturn = False #No overlap as default

        # for posToCheck in activeList:
        if len(ind) == 2:
            x,y = np.meshgrid( pts, pts )
            mask = x**2 + y**2 < ms**2
            curGrid = self.diskGrid[ ind[0]-ms:ind[0]+ms+1,\
                ind[1]-ms:ind[1]+ms+1]
        elif len(ind) == 3:
            x,y,z = np.meshgrid( pts, pts, pts )
            mask = x**2 + y**2 + z**3 < ms**3
            curGrid = self.diskGrid[ ind[0]-ms:ind[0]+ms+1,ind[1]-ms:ind[1]+\
                ms+1, ind[2] - ms:ind[2]+ms + 1]

        #checks using self.diskGrid
        toReturn = toReturn or np.any( curGrid * mask )
        return toReturn

    def float_to_index(self, floatTuple):
        return np.round(floatTuple).astype('int')

    def get_positions(self):
        return np.array(self.positions)


class PolydisperseDiskCollection:
    """
    Same as above, but for a selection of disks that is polydisperse. 
        self.diskGrid: The grid of the disk positions; 0 if there is no disk 
            at the nearest pixel, 1 if it's at the nearest pixel. 
        self.positions: [N]-element nested list of the discs' positions
        self.radii:     [N]-element nested list of the discs' radii. 
    Right now only works for 2 and 3D arrays; could be generalized. 
    This is not a rigorous thing... if you run forever you'll always get the 
    tails of the distribution appearing at a random spot... but this will give
    something that is polydisperse and close. 
    """
    #Follows a modified Bridson algorithm; it's dart throwing 
    #but I'm storing the value of the 
    def __init__( self, gridshape, get_radii, pad, k=30, maxSepFactor=1.0):
        """
        gridshape: The size of the grid shape. Basically the bounds outside
            which the particles don't appear. 
        get_radii: Function pointer. get_radii() should return a radius for a 
            particle (random). 
        pad: The maximum size of the particle. This will be enforced internally
            since otherwise the code will crash. INT scalar. 
        k: The # of trials to run before ceasing trying to add a particle. 
        amxSepFactor: Float scalar. The size of the step to take. Not sure if 
            it'll do anything. 
        """
        if len(gridshape) > 3:
            raise AttributeError("Can only handle 2D / 3D grids")

        #Step 0. Initialize:
        self.pad = pad
        self.diskGrid = np.zeros( np.array( gridshape) + 2*pad,dtype ='int')
        #We also need a counter for labeling particles:
        counter = 1
        
        #Step 1. Select an initial sample; initialize active list etc:
        startPos = np.random.rand( len( gridshape ) ) * np.array( gridshape )
        startInd = self.floatToIndex( startPos )
        self.diskGrid[tuple(startInd)] = counter
        #The active list will be the padded positions, the self.positions 
        #the raw positions. 
        activeList = [ [startPos + pad, 1*counter] ]
        counter += 1
        self.positions = [ startPos ]
        self.radii = [np.clip( get_radii(), 0, pad )]
        
        #Step 2: Loop over all elements in the active list:
        while len( activeList ) > 0:
            #1. Pick a random point in the list and see if we can add a particle
            #   near it:
            this_ind = random.randrange( len( activeList ) )
            curPos, part_lbl = activeList.pop( this_ind )

            #Popping it out; will put it back later
            #2. Generate up to k random points uniformly from the spherical 
            #   annulus on (minsep, 2*minsep); try to put them on:
            for a in xrange( k ): #with a continue statement if it works
                thisRadius= np.clip( get_radii(), 0, pad )
                minSep = 0.5*(thisRadius + self.radii[part_lbl-1] )
                thisPoint = np.random.randn( len( self.diskGrid.shape ) )
                thisPoint /= np.sqrt( (thisPoint**2).sum() )#normalize

                #Setting to correct magnitude:
                thisPoint *= minSep*(1.0 + maxSepFactor*np.random.rand() )
                thisPoint += curPos #the correct offset
                overlapped = self.check_overlap( thisPoint, minSep)

                # add a particle
                if not overlapped:
                    #First, put pack the stuff we popped out:
                    activeList.insert( this_ind, [curPos, part_lbl] )
                    activeList.append( [thisPoint, counter] )

                    #Adding the radii, de-padded position to the cum list:
                    self.positions.append( thisPoint - pad )
                    self.radii.append( thisRadius )
                    self.diskGrid[tuple(self.floatToIndex(thisPoint))]=counter
                    counter += 1
                    continue #to break out of the for loop

        #And I need to un-pad...
        self.diskGrid = self.diskGrid[minSep:-minSep, minSep:-minSep]

    def check_overlap( self, posToCheck, minSep ):
        ms = int( np.ceil( minSep ) ) 

        #ceil is needed... there is a lot of slop here...
        ind = self.floatToIndex( posToCheck )
        if np.any( ind <= self.pad ) or np.any( ind >= np.array( \
            self.diskGrid.shape) - self.pad ):
            return True
            #^checking if it's outside the image as an overlap

        pts = np.arange( -ms, ms+1)
        toReturn = False #No overlap as default
        # for posToCheck in activeList:
        if len( ind ) == 2:
            #2D
            x,y = np.meshgrid( pts, pts )
            mask = x**2 + y**2 < ms**2
            curGrid = self.diskGrid[ ind[0]-ms:ind[0]+ms+1,\
                ind[1]-ms:ind[1]+ms+1]
        elif len( ind ) == 3:
            #3D
            x,y,z = np.meshgrid( pts, pts, pts )
            mask = x**2 + y**2 + z**3 < ms**3
            curGrid = self.diskGrid[ ind[0]-ms:ind[0]+ms+1,ind[1]-ms:ind[1]+\
                ms+1, ind[2] - ms:ind[2]+ms + 1]

        #checks using self.diskGrid
        try:
            toReturn = toReturn or np.any( curGrid * mask )
        except:
            print curGrid.shape, mask.shape
            print posToCheck
            print ms
            print self.diskGrid.shape
            raise IOError
        return toReturn
            
    def floatToIndex( self, floatTuple ):
        ind = np.round( floatTuple).astype('int')
        return ind
