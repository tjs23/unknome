# Flywheel_AlignFunctions.py

Contains functions for the setup of FlyWheel images; detection of plate edges, wells, image alignment etc.

## Notable functions

### findTransformECC()

Performs the ECC transform, using OpenCV, to align flywheel images with their first, reference frame.

### findWellCentres

Takes a plate image and extracts contours corresponding to wells. Returns a list of the coordinates of the centres of the contours and a resized image whose histogram was equalised.

### estimateRowLocations

Takes well centre coordinates, clusters them in rows and fits linear regression lines to clusters to returns a sequence of line fits for the rows.

### findRowColBestfits

Takes a sequence of the plat rows linefits and returns the row and column bestfits. Also outputs intersects of the best-fits at the edges and centre of the plate image.

### calculateHomographyVertices

Takes a plate image and a list object containing m and b parameters of row and column bestfits and respective intersect datapoints at the edges and centre of the image. Returns a list object containing the coordinates of the homography vertices of the plate; homography vertices: H12, H1, A12 and A1 well's centres. Also returns the location of the row and column bestfits
        
### estimateWellsAH

Estimate well positions and plate corners from homography vertices.
        
# Flywheel_TrackFunctions.py

Contains functions to analyse the FlyWheel images to track fly motion. Plotting functions for the derived tracking data.

## Notable functions

### plateFlytracker()

Tracks flies in plate images using intensity contours after background subtration.

### plateToDCrawler()

Estimates time of death within plate wells; when movement consistently ceases. 
