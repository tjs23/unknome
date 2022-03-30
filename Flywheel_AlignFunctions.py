import os
import math
import sys
import cv2
from scipy import stats
import numpy as np

     
                
class AlignImageSeq():
    
    def __init__(self, assaydir):
        self.cwd = os.getcwd()
        self.fwdir = '%s\Dropbox\Unknome\Screens\Flywheel\PyWheel' %self.cwd
        self.workdir = os.path.join(self.fwdir, 'FilesOut')
        self.pickledir = os.path.join(self.fwdir, 'PickleFiles')
        self.assaydir = assaydir
        self.fwId = os.path.split(self.assaydir)[1]
        self.masterplatepath = os.path.join(self.fwdir, 'MasterPlate.jpg')
        self.masterplate = cv2.imread(self.masterplatepath, 0)
        self.masterplateMap = self.masterplateMap()
        self.mplateCorners = self.masterplateCorners()
        self.counter = 0
        self.recursion = 0
        self.xy_dim = (960, 540)
        self.rc_dim = self.xy_dim[::-1]
        self.frameZero = np.zeros(self.rc_dim, np.uint8)
        self.plateROI_def = [51, 408, 178, 608]
        self.frameDropouts = self.loadFrameDropouts()
        self.platesGeoMap = self.loadPlatesGeoMap()
        return
        
    def masterplateMap(self):
        '''It loads an ordered dictionary object: keys, sequence of integers from 1-96; values, tuple(x,y): well coordinates.
        Wells are numbered from top left corner (1, H12) to bottom right corner (96, A1)'''
        import cPickle as pickle
        picklepath = os.path.join(self.pickledir, 'masterplateMap.pickle')
        #load dictionary
        with open(picklepath, 'rb') as f:
            masterplateMap = pickle.load(f)
        return masterplateMap
    
    def masterplateCorners(self):
        mplateCorners = [np.asarray(self.masterplateMap[key]) for key in [1,12,85,96]]
        return mplateCorners
    
    def loadFrameDropouts(self):
        import cPickle as pickle
        picklepath = os.path.join(self.pickledir, 'frameDropouts.pickle')
        #load dictionary
        try:
            with open(picklepath, 'rb') as f:
                frameDropouts = pickle.load(f)
        except IOError:
            frameDropouts = {}
        return frameDropouts
        
    def angle_between(self, p1, p2):
        '''It takes two points and returns the angle in degrees between that line and 
        the horizontal axis.'''
        (x1, y1) = p1
        (x2, y2) = p2
        angle = np.rad2deg(np.arctan2(y2-y1, x2-x1))
        return angle 
                                       
    def auto_canny(self, image, sigma=0.33):
   	# compute the median of the single channel pixel intensities
   	v = np.median(image)
   	# apply automatic Canny edge detection using the computed median
   	lower = int(max(0, (1.0 - sigma) * v))
   	upper = int(min(255, (1.0 + sigma) * v))
   	edged = cv2.Canny(image, lower, upper)
   	return edged
   	
    def findTransformECC(self, image, frameZero, termination_eps, warp_mode = cv2.MOTION_EUCLIDEAN):
        '''It applies an ECC transform to an image, returns the transformed image. Keyword warp_mode defines 
        the motion mode of the transformation. The reference image is self.frameZero.'''
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)    
        # Specify the number of iterations.
        number_of_iterations = 10000
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = termination_eps;
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC (frameZero, image, warp_matrix, warp_mode, criteria)
        yrow, xcol = frameZero.shape
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            im_aligned = cv2.warpPerspective (image, warp_matrix, self.xy_dim, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im_aligned = cv2.warpAffine(image, warp_matrix, self.xy_dim, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        return im_aligned
    
    
    def eccGenerator(self):
        from Flywheel_TrackFunctions import FwObjects    
        #fetch and sort image sequence list
        filenamelist, imgpaths  = FwObjects().loadPlateImgSeq(mode = 'align', custompath = self.assaydir)
        #align plate images
        counter = 0
        for i, imgpath in enumerate(imgpaths):
            counter+=1; print('frame%s' %counter)
            filename = filenamelist[i]
            image = cv2.imread(imgpath, 0)#read image
            output = cv2.resize(image, self.xy_dim, interpolation = cv2.INTER_AREA)
            #apply ECC algorithm to align plate
            if counter == 1: 
                frameZero = self.masterplate
                #frameZero = output
                output_aligned = self.findTransformECC(output, frameZero, 1e-9)
                yield output_aligned, filename
                #yield output, filename
                frameZero = output_aligned
            else:
                try:
                    output_aligned = self.findTransformECC(output, frameZero, 1e-5)
                    yield output_aligned, filename
                    frameZero = output_aligned
                except:
                    print('ECC transform did not converge: frame%s was dropped.' %counter)
                    continue   
        return
        
                
    def findWellCentres(self, image, showcnt = False):
        '''It takes a plate image and extracts contours corresponding to wells. It returns a list object containing the coordinates
        of the centres of the contours (x, y) and a resized image whose histogram was equalised.'''
        if self.recursion == 0:
            # create a CLAHE object: equalise histogram adaptatively
            output = image.copy()
            clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
            output = clahe.apply(output)
            #resize image
            output = cv2.resize(output, self.xy_dim, interpolation = cv2.INTER_AREA)
        else:
            output = image
        #find edges
        edged = self.auto_canny(output, sigma=0.33)
        #find contours
        im2, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #filter out contours: draw bounding rectangles and circles
        wellCentres = []
        for cnt in cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if 10 < w <= 40 and 30 < h <= w:#filter bounding rectangles on size 
                (x0, y0) = (int(x+w/2.0), int(y+h/2.0))#rectangle center
                wellCentres.append((x0, y0))
                if showcnt:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.circle(output, (x0, y0), 2, (0, 255, 0), 2)#draw center     
        #filter contours on area
        cnts_filtered = [cnt for cnt in cnts if 600 <= cv2.contourArea(cnt) < 1150]
        #draw bounding circles on filtered contours
        boundingcircles = [cv2.minEnclosingCircle(cnt) for cnt in cnts_filtered]
        for circle in boundingcircles:
            (x,y) , radius = circle
            center = (int(x),int(y))
            radius = int(radius)
            if radius < 22:#filter bounding circles on size of radius
                wellCentres.append(center)
                if showcnt:
                    cv2.circle(output, center, radius, (0,255,0), 2)#draw circle                                
        #order wells in rows (y-coordinate)
        wellCentres = sorted(wellCentres, key = lambda x:x[1])
        wellCentres = [centre for centre in wellCentres if centre[0] > 150]#filter out centres outside plate
        #if no well centres were found, save and open output image, and repeat routine
        if len(wellCentres) == 0 and self.recursion <= 1:
            self.recursion +=1
            imgdirname = os.path.split(self.assaydir)[1]
            stockdir = os.path.join(self.workdir, imgdirname)
            if not os.path.exists(stockdir):
                os.mkdir(stockdir)
            filepath = os.path.join(stockdir, 'tempfile.jpg')
            cv2.imwrite(filepath, output)
            output = cv2.imread(filepath, 0)
            os.remove(filepath)
            wellCentres, output = self.findWellCentres(output, showcnt = False)
        return wellCentres, output
    
    
    def estimateRowLocations(self, wellCentres):
        '''It takes a sequence of tuples - well's centre coordinates (x,y) - clusters them in
        rows and fits linear regression lines to clusters. Returns a sequence of line fits; each fit includes: 
        slope, intercept, r-squared, number of datapoints and datapoints'''
        linefits = []; linepoints_rows = []
        for i, center in enumerate(wellCentres):
            #first well in the row 
            if len(linepoints_rows) == 0:
                linepoints_rows.append(center)
                continue
            #centre is in the same row
            elif abs(linepoints_rows[0][1]-center[1]) <= 25:
                linepoints_rows.append(center)
                #last element of the centre list
                if i+1 == len(wellCentres):
                    #filter linepoints that are beyond |0.5*radius| of the well radius
                    linepoints_rows = sorted(linepoints_rows, key = lambda x: x[1])
                    linepoints_rows_median = np.median([y for (x,y) in linepoints_rows])
                    linepoints_rows = [(x,y) for (x,y) in linepoints_rows if abs(linepoints_rows_median-y) <= 10]
                    try: 
                        x, y = zip(*linepoints_rows)
                    except ValueError:
                        continue
                    m, b, r_value, p, serr = stats.linregress(x, y)#fit regression line 
                    linefits.append((m, b, r_value**2, len(linepoints_rows), linepoints_rows))
                continue
            #centre belongs to a new row         
            elif abs(linepoints_rows[0][1] - center[1]) > 25:
                #test whether points are in the same well
                if len(linepoints_rows) == 2:
                    distance = np.sqrt((linepoints_rows[1][1]- linepoints_rows[0][1])**2 + (linepoints_rows[1][0]- linepoints_rows[0][0])**2)
                    if distance <= 20:
                        linepoints_rows = []
                        linepoints_rows.append(center)
                        continue
                #fit regression line        
                if len(linepoints_rows) > 2:
                    #filter linepoints that are beyond [-0.5, 0.5] of the well radius
                    linepoints_rows = sorted(linepoints_rows, key = lambda x: x[1])
                    linepoints_rows_median = np.median([y for (x,y) in linepoints_rows])
                    linepoints_rows = [(x,y) for (x,y) in linepoints_rows if abs(linepoints_rows_median-y) <= 10] 
                    x, y = zip(*linepoints_rows)
                    m, b, r_value, p, serr = stats.linregress(x, y)#fit regression line  
                    linefits.append((m, b, r_value**2, len(linepoints_rows), linepoints_rows))
                #test whether centre is sufficiently distant to belong to a new row 
                if abs(linepoints_rows[0][1] - center[1]) > 30:
                    linepoints_rows = []
                    linepoints_rows.append(center)
                    continue
                linepoints_rows = []
        #sort line fits on r-squared                 
        linefits = sorted(linefits, key = lambda x: x[2])
        return linefits
    
    
    def findRowColBestfits(self, linefits):
        '''It takes a sequence of the plate's rows linefits and returns the row and column bestfits. 
        Output is a list of tuples containing m and b parameters, and datapoints intersects of the bestfits
        at the edges and centre of the plate image.'''
        from itertools import dropwhile
        #pick row line best fit
        try:
            assert(len([row for row in linefits if row[-2] > 2]) > 0), 'frame%s yields less than 3 contours per row' %self.counter
            linefit_rows = [row for row in linefits if row[-2] > 2]#filter out linefits with less than 3 datapoints
            thresholds = np.arange(0.95, 0, -0.025)
            linefit_rows = [[row for row in linefit_rows if row[2] > threshold] for threshold in thresholds]#cluster linefits above r-squared threshold
            linefit_rows = list(dropwhile(lambda x:len(x) == 0, linefit_rows))#filter out empty clusters
            assert(len(linefit_rows)>0), 'frame%s yields no linefits' %self.counter 
            linefit_rows = sorted(linefit_rows[0], key = lambda x: x[-2])#sort highest r-squared cluster on number of datapoints per linefit
            bestfit_row = linefit_rows[-1]       
        except AssertionError, e:
            print(e); return None
        #row line best fit
        m_r, b_r, r2, size, data_row = bestfit_row
        #calculate row intersect datapoints
        try:
            xintersects = [0, 480, 960]
            yfit_intersects = [int(y) for y in np.polyval((m_r,b_r), xintersects)]
        except ValueError:
            return None
        p0_r, pm_r, p1_r = zip(xintersects, yfit_intersects)
        #calculate column bestfit
        xdata, ydata = zip(*data_row)
        yfit = [int(y) for y in np.polyval((m_r,b_r), xdata)]
        residuals = np.arange(0, 2, 0.05)
        rowcol_intersects = [[data_row[i] for i, y in enumerate(ydata) if abs(y-yfit[i]) <= residual] for residual in residuals]
        rowcol_intersects = list(dropwhile(lambda x:len(x)==0, rowcol_intersects))
        assert(len(rowcol_intersects)>0), 'No row-column intersects were found.'
        m_c = -1/m_r
        b_c = rowcol_intersects[0][0][1] - m_c * rowcol_intersects[0][0][0]
        #calculate column intersect datapoints
        try:
            yintersects = [0, 270, 540]
            xfit = [int(abs((val-b_c)/float(m_c))) for val in yintersects]
        except ValueError:
            return None
        p0_c, pm_c, p1_c = zip(xfit, yintersects)
        rowcol_intersects = [((m_r, b_r), (p0_r, pm_r, p1_r)), ((m_c, b_c), (p0_c, pm_c, p1_c))]
        return rowcol_intersects
        
    
    def calculateHomographyVertices(self, image, rowcol_intersects):
        '''It takes a plate image and a list object containing m and b parameters of row and column bestfits and 
        respective intersect datapoints at the edges and centre of the image. Returns a list object containing the coordinates of the 
        homography vertices of the plate; homography vertices: H12, H1, A12 and A1 well's centres. It returns also the location of the 
        row and column bestfits.'''
        #unpack
        [((m_r, b_r), (p0_r, pm_r, p1_r)), ((m_c, b_c), (p0_c, pm_c, p1_c))] = rowcol_intersects
        #identify column
        xm_c, ym_c = pm_c
        if xm_c >= 480:
            c_loc = 6 -(480-(xm_c+20))/50
        else:
            c_loc = 6 -(480-(xm_c-20))/50
        #fetch estimates of rows locations
        rowCoordinates = Maxima().fetchRowCoordinates(image, showfits = False, showmax = False)
        coordinates_values = rowCoordinates.values()
        row_midpoints = zip(*coordinates_values)[1]
        yrow_midpoints = zip(*row_midpoints)[1]
        yrow_midpoints = np.arange(80, 480, 50)
        #identify row
        r_loc = sorted([(i, abs(ymp - pm_r[1])) for i, ymp in enumerate(yrow_midpoints)], key = lambda x: x[1])[0]
        r_loc = rowCoordinates.keys()[r_loc[0]]
        #row and column identity
        rowcol_loc = [r_loc, c_loc]
        #calculate b-intersects of top and bottom rows
        b_r0 = b_r - ((r_loc-1)*49)
        b_r8 = b_r + ((8-r_loc) * 49)
        #calculate b-intersects of first and last column
        b_c0 =  m_c * (49 * (c_loc-1)) + b_c
        b_c12 = m_c * (-49 * (12 - c_loc)) + b_c
        #calculate rectangle vertices
        v_x0 = int((b_r0 - b_c0)/(m_r + m_c)); v_y0 = int(m_r * v_x0 + b_r0)
        v_x1 = int((b_r0 - b_c12)/(m_r + m_c)); v_y1 = int(m_r * v_x1 + b_r0)
        v_x2 = int((b_r8 - b_c0)/(m_r + m_c)); v_y2 = int(m_r * v_x2 + b_r8)
        v_x3 = int((b_r8 - b_c12)/(m_r + m_c)); v_y3 =  int(m_r * v_x3 + b_r8)
        #vertices
        p_v0 = (v_x0, v_y0);  p_v1 = (v_x1, v_y1); p_v2 = (v_x2, v_y2); p_v3 = (v_x3, v_y3)
        vertices_h = [p_v0, p_v1, p_v2, p_v3]
        return vertices_h, rowcol_loc
        
    
    def detectWellsAH(self, image):
        #find wells' centers
        wellcentres, output_img = self.findWellCentres(image)
        #estimate row locations
        rowfits = self.estimateRowLocations(wellcentres)
        #pick row and column bestfits
        rowcol_intersects = self.findRowColBestfits(rowfits)
        assert (rowcol_intersects is not None), 'frame yields no line fits'
        [((m_r, b_r), (p0_r, pm_r, p1_r)), ((m_c, b_c), (p0_c, pm_c, p1_c))] = rowcol_intersects#unpack
        angle = np.arctan2(p1_c[1]-p0_c[1], p1_c[0]-p0_c[0]) * 180 / np.pi 
        assert(abs(m_r + m_c)>0), 'OverflowError: cannot convert float infinity to integer'
        #estimate homography vertices
        vertices_h, rowcol_loc =  self.calculateHomographyVertices(image, rowcol_intersects)
        return vertices_h, angle
    
    
    def estimateWellsAH(self, image, showcnt = False):
        #invert image
        output_inv = 255 - image
        #threshold image dinamically
        for lowbound in xrange(230, 250):
            ret,thresh1 = cv2.threshold(output_inv,lowbound,255, cv2.THRESH_BINARY)
            output_copy, cnts, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnts_filtered = [cnt for cnt in cnts if 600 <= cv2.contourArea(cnt) < 1150]
            boundingcircles = [cv2.minEnclosingCircle(cnt) for cnt in cnts_filtered]
            boundingcircles = [circle for circle in boundingcircles if circle[0][0]<200]
            if len(boundingcircles) == 2:
                break
        assert len(boundingcircles) == 2, 'No contours were detected.'
        #calculate angle of joining line 
        datapoints, _ = zip(*boundingcircles)
        xcoord, ycoord = zip(*datapoints)
        angle = np.arctan2(ycoord[0]-ycoord[1], xcoord[0]-xcoord[1]) * 180 / np.pi 
        #define well displacements   
        dxy = np.asarray([(110, -20), (110, 25), (655, -20), (658, 26)])#displacements
        #estimate positions of wells on corners          
        boundingcircles = boundingcircles*2
        vertices = [np.asarray(c)+dxy[j] for j, (c,r) in enumerate(boundingcircles)]
        vertices = sorted(vertices, key = lambda x:x[1])
        if showcnt:
            for vertice in vertices:
                x,y = vertice
                center = (int(x),int(y))
                radius = 21
                cv2.circle(image, center, radius, (255,255,255), 1)#draw circle
            cv2.imshow('Contours', image)
            cv2.waitKey(0)        
        return vertices, angle
        
    
    def findPlateCorners(self, iternumb = 20, showells = False):
        from statsFunctions import GeometricMean
        import cPickle as pickle
        verticesList = []; angleList = []; counter = 0
        for image, filename in self.eccGenerator():
            counter += 1
            try:
                vertices, angle = self.estimateWellsAH(image)
                angleList.append(angle)
                verticesList.append(vertices)
            except AssertionError, e:
                print(e)
                try:
                    vertices, angle = self.detectWellsAH(image)
                    angleList.append(angle)
                    verticesList.append(vertices)
                except AssertionError, e:
                    print(e)
                    continue
            if len(verticesList) >= iternumb:
                plateCorners = [GeometricMean().calculateGeoMean(verticeset) for verticeset in zip(*verticesList)]
                if showells:
                    [GeometricMean().plotGeoMean(verticeset) for verticeset in zip(*verticesList)] 
                    for vert in plateCorners:
                        x,y = vert
                        center = (int(x),int(y))
                        radius = 21
                        cv2.circle(image, center, radius, (255,255,255), 1)#draw circle
                    cv2.imshow('vertices', image)
                    cv2.waitKey(0)
                break   
            elif counter/iternumb > 5:
                print('The positions of the plate corners could not be estimated.\n')
                plateCorners = None
                break
        #plate geopositional parameters
        plateGeoDef = [plateCorners, verticesList, angleList]
        #add plate geo definitions dictionary
        self.platesGeoMap[self.fwId] = plateGeoDef
        #save serialised dictionary
        picklepath = os.path.join(self.pickledir, 'platesGeoMap.pickle')
        with open(picklepath, 'wb') as f:
            pickle.dump(self.platesGeoMap, f, protocol = 2)               
        return plateGeoDef

    
    def loadPlatesGeoMap(self):
        '''Loads a dictionary object containing plate geo-positional definitions; key: flywheel identifier; 
        value: plateCorners, verticesList, angleList.'''
        import cPickle as pickle
        picklepath = os.path.join(self.pickledir, 'platesGeoMap.pickle')
        try:
            with open(picklepath, 'rb') as f:
                platesGeoMap = pickle.load(f)
        except IOError:
            platesGeoMap = {}
        return platesGeoMap
        
        
    def estimateAlignmentAngle(self, image):
        '''It returns an estimate of the angle of alignment of the plate in relation to the image'''
        try:
            vertices, angle = self.detectWellsAH(image) 
        except AssertionError, e:
            vertices, angle = self.estimateWellsAH(image)
        return vertices, angle
        
    def isPlateAligned(self, angle):
        '''Tests whether the plate is aligned in relation . It returns a boolean variable.'''
        from scipy.stats import norm
        #load plates geopositional dictionary
        platecorners, verticesList, angleList = self.platesGeoMap[self.fwId]
        center = np.median(angleList)
        madn =  np.median(abs(np.asarray(angleList) - center))/0.6745
        isaligned = norm(center, madn).pdf(angle) >= 0.05
        return isaligned
        
    
    def fetchPlateROI(self, image):
        '''It crops a flywheel plate image and returns the plate ROI.'''
        #mask definition 
        mask = np.zeros(self.rc_dim, np.uint8)
        [cv2.circle(mask, (int(x), int(y)), 21, 255, -1) for (x,y) in self.masterplateMap.values()]
        #invert mask
        mask_inv = cv2.bitwise_not(mask)
        #extract foreground: wells
        plateROI_fg = cv2.bitwise_and(image, image, mask=mask)
        #add foreground to background
        plateROI = cv2.add(plateROI_fg, mask_inv)
        #crop plate ROI
        row, h, col, w = self.plateROI_def
        plateROI = plateROI[row:row+h, col:col+w]
        return plateROI
        
    
    def alignImageSeq(self):
        import cPickle as pickle
        #path definition
        imgseqdir = os.path.join(self.assaydir, 'Imgseq')
        droplist = []
        #fetch plate positional parameters
        try:
            plateGeoDef = self.platesGeoMap[self.fwId]
        except KeyError:
            print('Estimating plate alignment angles distribution.')
            plateGeoDef = self.findPlateCorners()
        #unpack plate geodefinitions
        plateCorners, verticesList, angleList = plateGeoDef
        print('Aligning %s image sequence.' %self.fwId)
        for output_aligned, filename in self.eccGenerator():
            #variables definitions
            self.counter +=1; filepath = os.path.join(imgseqdir, filename)
            #estimate angle of alignment
            try:
                angle = self.estimateAlignmentAngle(output_aligned)
                #test whether plate is aligned if not try to re-align it
                if not self.isPlateAligned(angle): 
                    if self.counter == 1:
                        frameZero = self.masterplate
                    output_aligned = self.findTransformECC(output_aligned, frameZero, 1e-6, warp_mode = cv2.MOTION_EUCLIDEAN)
                    raise UserWarning('Frame%s may not be properly aligned.' %self.counter)
            except (AssertionError, UserWarning):
                #if angle could not be estimated save aligned image anyway 
                cv2.imwrite(filepath, output_aligned)
                droplist.append(filename)
                continue
            frameZero = output_aligned   
            #save aligned frame
            cv2.imwrite(filepath, output_aligned)
        #add droplist to dictionary        
        if len(droplist)>0:
            self.frameDropouts[self.fwId] = droplist
            picklepath = os.path.join(self.pickledir, 'frameDropouts.pickle')
            with open(picklepath, 'wb') as f:
                pickle.dump(self.frameDropouts, f, protocol = 2)  
        return  
        
            
    def batchAlignImageSeq(self, imgdir):
        '''It takes a directory path and aligns the image sequences of each assay in the directory.'''
        fwidList = os.listdir(imgdir)
        for fwid in fwidList:
            #reset variables
            self.assaydir = os.path.join(imgdir, fwid)
            self.fwId = os.path.split(self.assaydir)[1]
            self.counter = 0
            #align image sequence
            try:
                self.alignImageSeq()
            except:
                print('%s failed to align.' %self.fwId)
                continue
        return
    
    
    def surveyImgAlignment(self, dirpath):
        from Flywheel_TrackFunctions import PlateTracker
        import cv2
        fwIds = os.listdir(dirpath)
        for fwId in fwIds:
            print(fwId)
            try:
                filenamelist, pathlist = PlateTracker(fwId).loadPlateImgSeq()
            except ValueError:
                continue 
            idxlist = [0, int(len(filenamelist)/2.0), -1]
            imagepaths = [pathlist[val] for val in idxlist]
            alignedFrames = [cv2.imread(filepath, 0) for filepath in imagepaths]
            for frame in alignedFrames:
                cv2.imshow('Aligned', frame)
                cv2.waitKey(0)
        return          
    
    
    def surveyFramezero(self, dirpath):
        from datetime import datetime    
        fwIds = os.listdir(dirpath)
        #Fetch current time and date
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        for fwId in fwIds:
            dirpath = os.path.join(dirpath, fwId)
            filenamelist = [fname for fname in os.listdir(dirpath) if fname.startswith('FW')]
            filenamelist = sorted(filenamelist, key = lambda x:(int(x.split('-')[-2]), int(x.split('-')[-1][:-4])))
            droplist = []
            for i, filename in enumerate(filenamelist[:12]):
                    timestamps_str = [name[:-4].split('-') for name in filenamelist[i:i+2]]
                    timestamps_str = [('%s-%s-%s' %(date[:4], date[4:6], date[6:]), '%s:%s:%s' %(time[:2], time[2:4], time[4:])) for (flywheel, slot, date, time) in timestamps_str]
                    timestamps_str = ['%s %s' %(date, time) for (date, time) in timestamps_str]
                    datetime1, datetime2 = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")  for timestamp in timestamps_str]
                    diff_datetime = datetime2-datetime1
                    diff_minutes = diff_datetime.days * 1440 + diff_datetime.seconds/60.0#
                    if diff_minutes > 90:
                        droplist.append((i, filenamelist[i:i+2]))
            if len(droplist)>0:
                imgseqdir = os.path.join(dirpath, 'ImgSeq')
                ext = os.listdir(imgseqdir)[0][-3:]
                framezero = os.path.join(imgseqdir, '%s.%s' %(droplist[0][1][0][:-4], ext))
                if os.path.exists(framezero):
                    print(fwId); print(droplist)
                    filepath = os.path.join(self.workdir, 'surveyFramezero_%s.txt' %time)
                    with open(filepath, 'w') as f:
                        f.write('%s\n' %fwId)           
        return



class Maxima():
    def __init__(self):
        self.xy_dim = (960, 540)
        return
    
    def estimatePlateCentroid(self, image, showcentroid = False):
        # remove noise
        img = cv2.GaussianBlur(image,(3,3),0)
        # convolute with proper kernel
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)#y
        datapoints = []
        width, height = sobely.shape
        for x in xrange(0, width, 1):
            for y in xrange(0, height, 1):
                if sobely[x,y] == 255:
                    datapoints.append((x,y))
        xset, yset = zip(*datapoints)
        #calculate image centroid
        ym = min(yset) + (max(yset) - min(yset))/2
        xm = min(xset)+ (max(xset) - min(xset))/2
        centroid = (int(ym),int(xm))
        if showcentroid:
            cv2.circle(img, centroid, 4, (255,255,255), 2)
            cv2.imshow('edges', sobely)
            cv2.imshow('centroid', img)
            cv2.waitKey(0)
        return centroid
        
    
    def fetchMaxCoordinates(self, image, xlim = 300, ylim = 190):
        '''It takes a flywheel plate image and returns a list object with the coordinates of the local maxima. '''
        import numpy as np
        from skimage.feature import peak_local_max
        from skimage import img_as_float
        from scipy import ndimage as ndi
        import cv2
        #blur and invert image
        output = cv2.medianBlur(image, 5 )
        output_inv = abs(255-output)
        im = img_as_float(output_inv)
        # image_max is the dilation of im with a 20*20 structuring element
        # It is used within peak_local_max function
        image_max = ndi.maximum_filter(im, size = 20, mode = 'constant')
        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(im, min_distance = 20)
        #estimate plate centroid
        centroid = self.estimatePlateCentroid(image, showcentroid = False)
        Cx, Cy = centroid
        #filter coordinates on distance to centre of the image
        coordinates = [(y,x) for (y,x) in coordinates if abs(Cy-y) <= ylim and abs(480-x) <= xlim]
        #filter coordinates on absolute difference between maximum and mean intensity of neighborhood
        coordinates = [(y, x) for (y, x) in coordinates if abs(np.mean(output_inv[y-10:y+10, x-10:x+10])-output_inv[(y,x)]) >= 60]
        #filter out maxima that lie within the same neighborhood
        coordinates_filtered = []
        for (y, x) in coordinates:
            if np.amax(output_inv[y-10:y+10, x-10:x+10]) == output_inv[(y,x)]:
                maximum = output_inv[(y,x)]
                output_inv[y-10:y+10, x-10:x+10] = 0
                output_inv[(y,x)] = maximum
                coordinates_filtered.append((y,x))
        coordinates = sorted(coordinates_filtered, key = lambda x: (x[0], x[1]))#sort coordinates on rows
        return coordinates
    
                    
    def bestfitFromMaxCoordinates(self, maxCoordinates):
        '''It takes a list of the plate's maxima coordinates, clusters maxima detected in the same row, 
        fits regression lines to the clusters and returns the bestfit. '''
        import numpy as np
        from scipy import stats
        #define variables
        maxnumber = len(maxCoordinates)
        rowbounds = [int(val) for val in np.linspace(0, maxnumber, 9)]; #parse maxima per row
        #fit regression lines to maxima coordinates 
        lineFits = []
        for i, bound in enumerate(rowbounds[1:]):
            data = maxCoordinates[rowbounds[i]:bound]
            ydata , xdata = zip(*data)
            ymedian = np.median(ydata)
            data = [(y,x) for (y, x) in data if abs(ymedian-y) <=15]
            if len(data) > 2:
                yset, xset = zip(*data)
                slope, intercept, r_value, p_value, std_err = stats.linregress(xset, yset)#fit regression line
                bestfit = (data, slope, intercept, r_value, p_value, std_err)
                lineFits.append((i, data, slope, intercept, (r_value)**2, p_value, std_err))
            else:
                continue
        lineFits = sorted(lineFits, key = lambda x:x[4]) 
        bestfit = lineFits[-1]
        return bestfit
    
    
    def fetchRowCoordinates(self, image, showfits = False, showmax = False):
        '''It takes a flywheel plate image and returns an ordered dictionary object containing estimates 
        of the plate's rows coordinates. Keywords control whether rows' fits or plate's maxima are displayed.'''
        from collections import OrderedDict
        import numpy as np
        import cv2                   
        #resize image
        output = image.copy()
        output = cv2.resize(output, self.xy_dim, interpolation = cv2.INTER_AREA)
        #detect local maxima and fetch their coordinates
        maxCoordinates = self.fetchMaxCoordinates(output)
        #pick best line fit
        bestfit =  self.bestfitFromMaxCoordinates(maxCoordinates)
        row, data, m, b, r2, p, serr = bestfit #unpack
        yset, xset = zip(*data)
        #calculate estimates of rows coordinates
        xintersects = [0, 540, 960]
        yfit = [int(val)-row*50 for val in np.polyval((m,b), xintersects)]
        linepoints = (px, pm, py) = zip(xintersects, yfit)
        rowCoordinates = [[(x, y + (i*50)) for (x,y) in linepoints] for i in xrange(8)]
        if showfits or showmax:
            #show maxima
            if showmax: 
                #draw maxima on image
                for coord in maxCoordinates:
                    y, x = tuple(coord)
                    cv2.circle(output, (x,y), 2, (255, 255, 255), 2)
            #draw regression lines
            if showfits:
                for row in rowCoordinates:
                    px, pm, py = row
                    cv2.line(output, px, py,(255,255,255), 1)
            cv2.imshow('rows', output)
            cv2.waitKey(0)
        #build dictionary
        rowCoordinates = OrderedDict(zip([i+1 for i in xrange(8)], rowCoordinates))
        return rowCoordinates


#AlignImageSeq().batchAlignImageSeq(imagedir)

    


