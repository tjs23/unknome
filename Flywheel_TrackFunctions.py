import sys
import os
import numpy as np
import pandas as pd

class Dashboard():
    
    def __init__(self):
        self.cwd = os.getcwd()
        self.dbdir = '%s\Dropbox\Unknome\Databases' %self.cwd
        self.dbpath = os.path.join(self.dbdir, 'FlywheelDB.db')
        drivepath = self.getDriveLetter('Flywheel\Data')
        self.basedir = drivepath
        #self.fwdir = '%s\Dropbox\Unknome\Screens\Flywheel\PyWheel' %self.cwd
        self.fwdir = 'U:\Flywheel\Data\OtherUsers\David\PyWheel'
        self.workdir = os.path.join(self.fwdir, 'FilesOut')
        self.pickledir = os.path.join(self.fwdir, 'PickleFiles')
        self.stockdatadir = os.path.join(self.pickledir, 'Stockdata')
        self.tracksdir = os.path.join(self.pickledir, 'PlateFlyTracks')
        
        dirpaths = [self.workdir, self.pickledir, self.stockdatadir, self.tracksdir]
        [os.makedirs(path) for path in dirpaths if not os.path.exists(path)]
        
        self.datalistspath = os.path.join(self.pickledir, 'plateDatalists.pickle')
        self.plateDatalists = self.loadDatalists()
        self.medSurv_dict = self.loadMedSurv()
        self.wellIds = self.wellIDs()
        self.ctrlnames = self.controlsDict().values()
        return
    
    def getDriveLetter(self, fileID):
        from string import ascii_uppercase
        drivepath = ''
        for letter in ascii_uppercase:
            drive = letter + ":\\"
            if os.path.exists(os.path.join(drive, fileID)):
                drivepath = os.path.join(drive, fileID)
                break
        try:
            assert (os.path.exists(drivepath)), 'The remote drive is not connected.\n'
        except AssertionError, e:
            print(e)
            drivepath = ''
        return drivepath
                
    def controlsDict(self):
        #Define dictionary
        controlsDict = {'EMPTY': 'Empty', 'GFPI': 'GFPi', 'W1118': 'w1118', 'CG11325': 'CG11325'} 
        return controlsDict
        
    def wellIDs(self):
        #Define wells labels
        rowLabels  = list(map(chr, reversed(range(ord('A'), ord('H')+1))))
        colNumbers = [i+1 for i in xrange(12)]
        wellIds = [('%s%i' %(label, number)) for label in rowLabels for number in colNumbers[::-1]]
        return wellIds
        
    def loadDatalists(self):
        import cPickle as pickle
        if os.path.exists(self.datalistspath):
            with open(self.datalistspath, 'rb') as f:
                plateDatalists = pickle.load(f)
        else:
            plateDatalists = {}
        return plateDatalists
        
    def loadMedSurv(self):
        import cPickle as pickle
        medsurvpath = os.path.join(self.pickledir, 'medSurv_dict.pickle')
        if os.path.exists(medsurvpath):
            with open(medsurvpath, 'rb') as f:
                medSurv_dict = pickle.load(f)
        else:
            medSurv_dict = {}
        return medSurv_dict 
              
    def batchDates(self):
        from datetime import datetime
        strdates = [['04082012', '17082012', '14092012', '05102012', '01032013', '13112015', '16122015'], ['25102012', '21112012', '19122012', '18012013', '21012013', '01032013', '17112015', '16122015']]
        tables = ['ROS', 'Starvation']
        batchdates = [[datetime.strptime(date, '%d%m%Y').date() for date in table] for table in strdates]
        batchdates = dict(zip(tables, batchdates)) 
        return batchdates
    
    
    def resetAssaydirStructure(self, dirpath, batchmode = True):
        from shutil import move
        #batch mode switch
        fwids = [os.path.split(dirpath)[1]]
        if batchmode:    
            fwids = os.listdir(dirpath)    
        #sweep through assay folders
        for fwid in fwids:
            print('updating %s folder' %fwid) 
            basedir = dirpath
            if batchmode:
                basedir = os.path.join(dirpath, fwid)
            #fetch image file list and move files to root directory    
            jpegfolder = os.path.join(basedir, 'jpeg')
            if os.path.exists(jpegfolder):
                imgfilelist = os.listdir(jpegfolder)
                imgfilepaths = [os.path.join(jpegfolder, filename) for filename in imgfilelist]
                [move(filepath, basedir) for filepath in imgfilepaths]
            else:
                print('%s folder does not exist') %jpegfolder
            #create ImgSeq folder
            imgseqdir = os.path.join(basedir, 'ImgSeq')
            if not os.path.exists(imgseqdir):
                os.makedirs(imgseqdir)
            #test whether avi folder exists and rename it
            avifolder = os.path.join(basedir, 'avi')
            if os.path.exists(avifolder):
                os.rename(avifolder, os.path.join(basedir, 'Movies'))#rename avi folder
            #avi folder does not exist, move avi files to movies folder
            elif len([filename for filename in os.listdir(basedir) if filename[-3:] == 'avi'])>0:
                avifilelist = [filename for filename in os.listdir(basedir) if filename[-3:] == 'avi']
                moviesfolder = os.path.join(basedir, 'Movies')
                if not os.path.exists(moviesfolder):
                    os.makedirs(moviesfolder)
                avifilepaths = [os.path.join(basedir, filename) for filename in avifilelist]
                [move(filepath, moviesfolder) for filepath in avifilepaths]
            #remove jpeg folder
            if os.path.exists(jpegfolder):    
                try:
                    assert(len(os.listdir(jpegfolder)) == 0), '%s folder is not empty and will not be deleted' %jpegfolder
                    try:
                        os.remove(jpegfolder)
                    except WindowsError:
                        print('%s folder could not be deleted' %jpegfolder)
                        pass   
                except AssertionError, e:
                    print(e)
                    continue
        return
    
    
    def filteroutBlankImages(self, dirpath, batchmode = True):
        from shutil import move
        from hurry.filesize import size
        #batch mode switch
        fwids = [os.path.split(dirpath)[1]]
        if batchmode:    
            fwids = os.listdir(dirpath)
        #sweep through assay folders
        for fwid in fwids:
            basedir = dirpath
            if batchmode:
                basedir = os.path.join(dirpath, fwid)
            print('filtering %s folder' %fwid)
            #fetch image list
            filepaths =[os.path.join(basedir, name) for name in os.listdir(basedir) if name[-4:] == '.jpg']
            filepaths_filtered = [path for path in filepaths if int(size(os.stat(path).st_size)[:-1]) < 85]#filter out on file size
            #move filtered out files to new folder
            newdir = os.path.join(basedir, 'EmptyFiles_bin')
            if not os.path.exists(newdir):
                os.makedirs(newdir)
            new_filepaths = [os.path.join(newdir, os.path.split(filepath)[1]) for filepath in filepaths_filtered]
            [move(filepath, new_filepaths[i]) for i, filepath in enumerate(filepaths_filtered)]#move filtered out files to a new folder
        return
    
    
    def findMisplacedFrames(self, dirpath, batchmode = False):
        from shutil import move
        from datetime import datetime
        #define variables
        fwIds = [os.path.split(dirpath)[1]]; assaydir = [dirpath]
        if batchmode:
            fwIds = [fwId for fwId in os.listdir(dirpath)]
            assaydir = [os.path.join(dirpath, fwId) for fwId in fwIds]  
        #find and move frames that are less than 2 minutes apart
        for j, fwId in enumerate(fwIds):        
            print('%s' %fwId)
            #fetch and sort image sequence list
            filenamelist, imgpaths  = FwObjects().loadPlateImgSeq(mode = 'align', custompath = assaydir[j])
            #parse datetime objects from file timestamps
            droplist = []
            for i, filename in enumerate(filenamelist[:-1]):
                timestamps_str = [name[:-4].split('-') for name in filenamelist[i:i+2]]
                timestamps_str = [('%s-%s-%s' %(date[:4], date[4:6], date[6:]), '%s:%s:%s' %(time[:2], time[2:4], time[4:])) for (flywheel, slot, date, time) in timestamps_str]
                timestamps_str = ['%s %s' %(date, time) for (date, time) in timestamps_str]
                datetime1, datetime2 = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")  for timestamp in timestamps_str]
                diff_datetime = datetime2-datetime1
                diff_minutes = diff_datetime.days * 1440 + diff_datetime.seconds/60.0#
                if diff_minutes < 2.0:
                    droplist.append(filename)
            #move files to a different folder        
            if len(droplist)>0:
                #create misplaced frames folder
                misframes_folder = os.path.join(assaydir[j], 'MisplacedFrames')
                if not os.path.exists(misframes_folder):
                    os.makedirs(misframes_folder)
                #move files into folder
                misframes_filepaths = [os.path.join(assaydir[j], filename) for filename in droplist]
                [move(filepath, misframes_folder) for filepath in misframes_filepaths]
        return 
        
    
    def updateTrackdataMap(self):
        import cPickle as pickle
        #fetch trackdata in trackdir
        trackdata_files = [fil for fil in os.listdir(self.tracksdir) if os.path.splitext(fil)[1] == '.pickle']
        picklepaths = [os.path.join(self.tracksdir, fil) for fil in trackdata_files]
        #parse fwIds from filenames
        fwIds = [os.path.splitext(fil)[0][:-10] for fil in trackdata_files]
        #build dict and pickle it
        trackdataDict = dict(zip(fwIds, picklepaths))
        trackdatapath = os.path.join(self.pickledir, 'trackdataPlatesMap.pickle')
        with open(trackdatapath, 'wb') as f:
            pickle.dump(trackdataDict, f, protocol = 2)
        return
        
    def updateDatalistsDict(self):
        import cPickle as pickle
        #fetch stockdata in stockdata dir
        stockdata_files = [fil for fil in os.listdir(self.stockdatadir) if os.path.splitext(fil)[1] == '.pickle']
        picklepaths = [os.path.join(self.stockdatadir, fil) for fil in stockdata_files]
        #parse fwIds from filenames
        fwIds = [os.path.splitext(fil)[0] for fil in stockdata_files]
        #build dict and pickle it
        datalistsDict = dict(zip(fwIds, picklepaths))
        with open(self.datalistspath, 'wb') as f:
            pickle.dump(datalistsDict, f, protocol = 2)
        return

               
    def updateMedSurvDict(self):
        import cPickle as pickle
        #fetch plateIDs from stockdata
        fwIDs = [os.path.splitext(name)[0] for name in os.listdir(self.stockdatadir)]
        #update dictionary
        medSurv_dict = {}
        for fwId in fwIDs:
            print(fwId)
            stockdata = DataVis(fwId).fetchStockdata()
            medsurv, error = PlateTracker(fwId).km_medSurv(stockdata, alpha = 0.01)
            medSurv_dict[fwId] = [medsurv, error]
        #pickle dictionary
        medsurvpath = os.path.join(self.pickledir, 'medSurv_dict.pickle')
        with open(medsurvpath, 'wb') as f:
            pickle.dump(medSurv_dict, f, protocol = 2)
        return       
    
    def updateDataDictionaries(self):
        self.updateTrackdataMap()
        print('Trackdatamap was updated.\n')
        self.updateDatalistsDict()
        print('Datatlists dictionary was updated.\n')
        print('Updating medians survival dictionary.\n')
        self.updateMedSurvDict()
        FwObjects().clusterPlateIDs()
        FwObjects().xywellMapBuilder() 
        FwObjects().masterplateMapBuilder()
        return
    
    def rebuildFlywheelDB(self, textfilepath):
        rows = FwheelDB().fetchRows(textfilepath)
        FwheelDB().createFlywheelDB(rows)
        FwheelDB().updateFwIds()
        return
    
    def batchPlateAnalyser(self, dirpath, period = 20, output = 'remote'):
        fwIds = os.listdir(dirpath)
        for fwId in fwIds:
            PlateTracker(fwId, custompath = dirpath).plateAnalyser(period = period, output = output)
        return



class FwheelDB(Dashboard):
     
    def __init__(self):
        Dashboard.__init__(self)
       
    
    def getMembers(self):
        import inspect
        methods = inspect.getmembers(self, predicate = inspect.ismethod)
        print(methods)
        return         

    def fetchRows(self, textpath):
        #fetch column headings
        with open(textpath, 'r') as f:
            colheads = f.readline()[:-1]
            colheads = colheads.split('\t')
        df = pd.read_csv(textpath, delimiter = '\t')
        data = [df[head] for head in colheads]
        rows = zip(*data)
        #convert datatypes
        rows = [(int(fwkey),b,c,d,e,f, int(f_age),sex, int(s),date,int(fwN[2:]),int(slotN),h,i,j,l) for (fwkey,b,c,d,e,f,f_age,sex,s,date,fwN,slotN,h,i,j,l) in rows]
        #cluster rows
        rows = [[row for row in rows if row[1] == 'Unknome'], [row for row in rows if row[1] == 'Other']]
        rows = [[row for row in rows[0] if row[2] == 'ROS'], [row for row in rows[0] if row[2] == 'Starvation'], rows[1]]
        return rows


    def createFwTable(self, tablename):
        import sqlite3
        #Connect to database
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        #create table
        print('Creating table %s. \n' %tablename)
        createStatement = '''CREATE TABLE  %s (sqlKey INTEGER PRIMARY KEY AUTOINCREMENT, fwKey INTEGER NOT NULL, Project TEXT NOT NULL, Assay CHAR(50) NOT NULL, Conditions CHAR(100) NOT NULL,
                            
                            StockID TEXT NOT NULL, Genotype CHAR(50) NOT NULL, Fly_age INT NOT NULL, Fly_sex TEXT NOT NULL, SampleN INT NOT NULL, Start_date DATE NOT NULL, 
                            
                            FwheelN INTEGER NOT NULL,  SlotN INTEGER NOT NULL, Comments CHAR(255), Validation TEXT NOT NULL, fwID TEXT NOT NULL, ImgAnalysis TEXT NOT NULL)''' %tablename
        cursor.execute(createStatement)
        return  
    
       
    def createFlywheelDB(self, rows):
        import sqlite3             
        #Connect to database
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        #define tablenames
        tablenames = ['ROS', 'Starvation', 'Other']   
        for i, tablename in enumerate(tablenames):
            try:    
                cursor.execute('''SELECT name FROM sqlite_sequence WHERE name = ?''', (tablename,))
                if len(cursor.fetchall()) == 0:
                    self.makeTable(tablename)          
                else:
                    print('Table %s already exists. \n' %tablename)  
            except:       
                self.createFwTable(tablename)
            insertStatement = '''INSERT INTO %s (fwKey, Project, Assay, Conditions, StockID, Genotype, Fly_age, Fly_sex, SampleN, Start_date, FwheelN, SlotN, 
                                Comments, Validation, FwID, ImgAnalysis) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''' %tablename 
            print('Inserting data in table %s. \n' %tablename)
            cursor.executemany(insertStatement, rows[i])
            db.commit()
            db.close()
            return
    
    
    def addRecordsToDB(self, filename, rowdelimiter, tablenames = ['ROS', 'Starvation', 'Other']):
        from itertools import ifilter
        import sqlite3 
        dirpath = 'C:\Users\Unknome\Dropbox\Unknome\Screens\Flywheel\Database'
        textpath = os.path.join(dirpath, filename)
        #fetch rows
        rows = self.fetchRows(textpath)
        rows_filtered = [list(ifilter(lambda x: x[0] > rowdelimiter, table)) for table in rows]#select rows to add
        #parse fwIds
        for j, table in enumerate(rows_filtered):
            for i, row in enumerate(table):
                b1, b2, b3 = row[14].split('_')
                b2 = ''.join(b2.split('/'))
                new_fwId = ('_').join([b1, b2, b3])
                new_row = list(row[:14])
                new_row.append(new_fwId); new_row.append(row[15])
                rows_filtered[j][i] = new_row
        #build dictionary    
        tabledataDict = dict(zip(['ROS', 'Starvation', 'Other'], rows_filtered))
        #Connect to database
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        #insert data   
        for tablename in tablenames:
            cursor.execute('''SELECT name FROM sqlite_sequence WHERE name = ?''', (tablename,))
            insertStatement = '''INSERT INTO %s (fwKey, Project, Assay, Conditions, StockID, Genotype, Fly_age, Fly_sex, SampleN, Start_date, FwheelN, SlotN, 
                                Comments, Validation, FwID, ImgAnalysis) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''' %tablename 
            if len(tabledataDict[tablename]) > 0:
                print('Inserting data in table %s. \n' %tablename)
                cursor.executemany(insertStatement, tabledataDict[tablename])
                db.commit()
        db.close()
        return
                

    def updateFwIds(self):
        import sqlite3
        #connect to database
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        #fetch tablenames from database
        cursor.execute('''SELECT name FROM sqlite_sequence''')
        tablenames = cursor.fetchall()
        #fetch fwIds from database
        data = []
        for tablename in tablenames:
            cursor.execute('''SELECT fwKey, fwId FROM %s''' %tablename)
            tabledata = cursor.fetchall()
            data.append(tabledata)
        #parse fwIds
        for table in data:
            for i, (a, b) in enumerate(table):
                [b1, b2, b3] = b.split('_')
                b2 = ''.join(b2.split('/'))
                b = ('_').join([b1, b2, b3])
                table[i] = (a, b)
        #zip data         
        data = [zip(*table) for table in data]
        #update database fwIds
        for i, table in enumerate(data):
            for j, fwkey in enumerate(table[0]):
                cursor.execute('''UPDATE %s SET fwID = ? WHERE fwKey = ?''' %tablenames[i], (table[1][j], fwkey)) 
        db.commit()
        db.close()
        return      
    
                  
    
class DatabaseOperations(Dashboard):
    
    def __init__(self):
        Dashboard.__init__(self)
        
    def fetchStockset(self):
        import sqlite3        
        #Connect to database
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        #fetch tablenames from database
        cursor.execute('''SELECT name FROM sqlite_sequence''')
        tablenames = cursor.fetchall()
        #fetch fwIds from database
        stockset = {}
        for tablename in tablenames:
            cursor.execute('''SELECT StockID FROM %s''' %tablename)
            tabledata = [tupl[0] for tupl in cursor.fetchall()]
            table_stockset = set(tabledata)
            stockset[tablename[0]] = table_stockset
        return stockset
        
     
    def fetchPlateSet(self):
        import sqlite3        
        #Connect to database
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        #fetch tablenames from database
        cursor.execute('''SELECT name FROM sqlite_sequence''')
        tablenames = cursor.fetchall()
        #fetch fwIds from database
        plateset = {}
        for tablename in tablenames:
            cursor.execute('''SELECT fwID FROM %s WHERE Validation = ?''' %tablename, ('OK',))
            tabledata = [tupl[0] for tupl in cursor.fetchall()]
            table_plateset = set(tabledata)
            plateset[tablename[0]] = table_plateset
        return plateset
        
        
    def fetchPlateIDFromStock(self, stockID, tablenames = ['ROS', 'Starvation']):
        import sqlite3       
        #test whether stockname is a control
        stockID = stockID.upper()
        try:
            controlsDict = self.controlsDict()
            stockID = controlsDict[stockID]
        except KeyError:
            pass   
        #Connect to database
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        #test whether tables exist
        if isinstance(tablenames, str):
            tablenames = [tablenames]
        for name in tablenames:
            cursor.execute('''SELECT * FROM sqlite_sequence WHERE name = ? ''', (name,))
            if len(cursor.fetchall()) == 0:
                print('NameError: %s is not a valid table name. Such table does not exist in the database. \n' %name)
                tablenames.remove(name)
                if len(tablenames) == 0:
                    print('Please, enter a valid table name and try again!')
                    sys.exit()
                else:
                    continue
        #Fetch data from fwDB tables
        data = []
        for name in tablenames:
            cursor.execute('''SELECT * FROM %s WHERE StockID = ? ''' %name , (stockID,))
            sublist = cursor.fetchall()
            data.append(sublist)
        #unpack data
        if len(tablenames) == 1:
            fwIds = [[tupl[15] for tupl in sublist] for sublist in data]
        elif len(tablenames) > 1:
            fwIds = [[(tupl[3], tupl[15]) for tupl in sublist] for sublist in data]
        fwIds = [tupl for sublist in fwIds for tupl in sublist]
        return fwIds
                
                                                                                                                                                                                                                                                                                                                                                                                                                      
    def fetchRecord(self, colval, colID, tablename = 'ROS'):
        import sqlite3
        #Connect to database
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        #Fetch data from areas table
        cursor.execute('''SELECT * FROM %s WHERE %s = ? ''' %(tablename, colID), (colval,))
        data = cursor.fetchall()
        return data
    
    def fetchRecSampleN(self, fwId):
        from itertools import chain
        u96Records = FwObjects().loadU96Records()
        u96Records_unpacked = list(chain(*u96Records.values()))
        recSampleN = [record[1] for record in u96Records_unpacked if record[0]==fwId][0]
        return recSampleN
    
    def fetchTobeDoneList(self, tablename = 'ROS'):
        from Unknome_Functions import Unknome
        from itertools import ifilter
        #load data lists
        stockset = self.fetchStockset()[tablename]
        viables = Unknome().fetchViables()
        #subtract lists and filter out controls
        tobedone = list(ifilter(lambda x: x not in viables, stockset))
        tobedone = list(ifilter(lambda x: x not in self.controlsDict().values() and x.startswith('JS'), tobedone))
        return tobedone
    
    def fetchRecordComments(self, screen = 'ROS'):
        import sqlite3
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        cursor.execute('''SELECT fwID, Comments FROM %s WHERE Comments IS NOT NULL''' %screen)
        reComments = cursor.fetchall()
        return reComments
    
    def fetchScreenControls(self, screen = 'ROS'):
        from itertools import ifilter
        #load stockset
        stockset = self.fetchStockset()[screen]
        cgnumberList = list(ifilter(lambda x:x.startswith('CG'), stockset))
        screenControls = self.ctrlnames + cgnumberList
        return screenControls

    def fetchBackgroundControls(self, screen = 'ROS'):
        recordComments =self.fetchRecordComments(tablename = screen)
        backgControls = [tupl[0] for tupl in recordComments if tupl[1].startswith('RNAi line')]
        return backgControls
        
                                                                                                                                                                                                                                                                                                                                                                                              


class FwObjects(Dashboard):
    
    def __init__(self):
        Dashboard.__init__(self)
        return
       
    def getMembers(self):
        import inspect
        methods = inspect.getmembers(self, predicate = inspect.ismethod)
        print(methods)
        return
           
    def clusterPlateIDs(self, tablename = 'ROS'):
        import cPickle as pickle
        from datetime import datetime
        #fetch table plateIDs
        plateset = DatabaseOperations().fetchPlateSet()
        plateId_ros = plateset[tablename]
        #fetch batchdates
        batchdatesDict = self.batchDates()
        batchdates = batchdatesDict[tablename]
        startdate = datetime.strptime('01012000', '%d%m%Y').date()
        batchdates.insert(0, startdate) 
        #fetch the full entry for each plateID
        records = [DatabaseOperations().fetchRecord(Id, 'fwID', tablename = tablename) for Id in plateId_ros]
        records = [record for lst in records for record in lst]#unpack
        #trim entry data
        records = [(record[1], record[15].split('_')[1], record[11], record[15]) for record in records]
        records = [(a, datetime.strptime(b, '%d%m%Y').date(), c, d) for (a,b,c,d) in records]
        #sort plateIds
        records = sorted(records, key = lambda x: (int(x[0]), int(x[2])))
        #cluster plateIDs: batches
        clusterRecords = [[record for record in records if date < record[1] <= batchdates[i+1]] for i, date in enumerate(batchdates[:-1])]
        #cluster plateIDs: wheels
        wheelN = [1, 2, 3]
        clusterRecords = [[[record[3] for record in batch if record[2] == number] for number in wheelN] for batch in clusterRecords]
        #build dictionary and pickle it
        clusterRecords = [dict(zip(wheelN, batch)) for batch in clusterRecords]
        batchkeys = ['batch%s' %str(j+1) for j in xrange(len(batchdates))]
        clusterPlateIDs = dict(zip(batchkeys, clusterRecords))
        clusterpath = os.path.join(self.pickledir, 'clusterPlateIDs_%s.pickle' %tablename)
        with open(clusterpath, 'wb') as f:
            pickle.dump(clusterPlateIDs, f, protocol = 2) 
        return


    def loadClusterPlateIDs(self, screen = 'ROS'):
        import cPickle as pickle
        clusterpath = os.path.join(self.pickledir, 'clusterPlateIDs_%s.pickle' %screen) 
        with open(clusterpath, 'rb') as f:
            clusterPlateIDs = pickle.load(f)
        return clusterPlateIDs
        
                                                                
    def xywellMapBuilder(self):
        import cPickle as pickle
        #definitions
        path = os.path.join(self.fwdir, 'wells_xy.txt')
        headlist = ['well_Id', 'X', 'Y']
        #fetch well coordinates
        df = pd.read_csv(path, delimiter = '\t')
        xycoord = [df[head] for head in headlist[1:]]
        xycoord = zip(*xycoord)
        #build dictionary
        wellIds = df[headlist[0]]
        xywellMap = dict(zip(wellIds, xycoord))
        #serialize dictionary
        picklepath = os.path.join(self.pickledir, 'xywellMap.pickle')
        with open(picklepath, 'wb') as f:
            pickle.dump(xywellMap, f, protocol = 2)     
        return
    
      
    def xywellMap(self):
        import cPickle as pickle
        picklepath = os.path.join(self.pickledir, 'xywellMap.pickle')
        #load dictionary
        with open(picklepath, 'r') as f:
            xywellMap = pickle.load(f)
        return xywellMap
        
    
    def masterplateMapBuilder(self):
        import cPickle as pickle
        from collections import OrderedDict
        #definitions
        path = os.path.join(self.fwdir, 'MasterPlate_wells_map.txt')
        headlist = ['Label', 'X', 'Y']
        #fetch well coordinates
        df = pd.read_csv(path, delimiter = '\t')
        xycoord = [df[head] for head in headlist[1:]]
        xycoord = zip(*xycoord)
        #build dictionary
        wellIds = df[headlist[0]]
        xywellMap = OrderedDict(zip(wellIds, xycoord))
        #serialize dictionary
        picklepath = os.path.join(self.pickledir, 'masterplateMap.pickle')
        with open(picklepath, 'wb') as f:
            pickle.dump(xywellMap, f, protocol = 2)     
        return
    
    def masterplateMap(self):
        import cPickle as pickle
        picklepath = os.path.join(self.pickledir, 'masterplateMap.pickle')
        #load dictionary
        with open(picklepath, 'rb') as f:
            masterplateMap = pickle.load(f)
        return masterplateMap
    
    def loadPlateFlytracks(self):
        import cPickle as pickle
        #load data
        trackdatamap = self.loadTrackdataPlatesMap()
        picklepath = trackdatamap[self.fwId]
        with open(picklepath, 'rb') as f:
            plateFlytracks = pickle.load(f)
        return plateFlytracks  
        
                            
    def loadTrackdataPlatesMap(self):
        import cPickle as pickle
        picklefile = os.path.join(self.pickledir, 'trackdataPlatesMap.pickle')
        try: 
            with open(picklefile, 'rb') as f:
                trackdataMap = pickle.load(f)
        except IOError:
            trackdataMap = {}
        return trackdataMap
    
    def loadStockdata(self):
        import cPickle as pickle
        try:
            stockpath = self.plateDatalists[self.fwId]
            with open(stockpath, 'rb') as f:
                stockdata = pickle.load(f)
        except KeyError:
            print('%s has either not been analysed or it is not a valid plate ID. \n' %self.fwId)
        return stockdata 
        
        
    def loadPlateImgSeq(self, mode = 'track', custompath = ''):
        #define paths to image sequences directory
        if mode == 'align':
            imgseqdir = os.path.join(custompath, 'ImgSeq')
        else:
            tablenames = ['ROS', 'Starvation', 'Dessication']
            rosplate_imgseq, starvplate_imgseq, dessplate_imgseq = [os.path.join(self.basedir, '%s\Assays\%s\ImgSeq' %(tablename, self.fwId)) for tablename in tablenames]
            #test whether path exists
            if os.path.exists(rosplate_imgseq):
                imgseqdir = rosplate_imgseq
            elif os.path.exists(starvplate_imgseq):
                imgseqdir = starvplate_imgseq
            elif os.path.exists(dessplate_imgseq):
                imgseqdir = dessplate_imgseq
            elif os.path.exists(self.custompath):
                imgseqdir = os.path.join(self.custompath, 'ImgSeq')
            else:
                raise ValueError('Plate %s could not be found. \n' %self.fwId)
        #fetch image filenames and paths
        switch = {'align': custompath, 'track': imgseqdir}    
        filenamelist = [filename for filename in os.listdir(switch[mode]) if filename.startswith('FW')]   
        try:
            fw, plate, date, time = filenamelist[0].split('-')
            assert (len(date) == 8 and len(time[:-4]) == 6)
            filenamelist = sorted(filenamelist, key = lambda x: (int(x.split('-')[-2]), int(x.split('-')[-1][:-4]))) #sort on timestamp
            self.timelabel = 'timestamp'
        except AssertionError:
            self.timelabel = 'framenumber'
            try:
                filenamelist = sorted(filenamelist, key = lambda x: int(x.split('-')[-1][:-4])) #sort on frame number
            except ValueError:
                filenamelist = sorted(filenamelist, key = lambda x: int(x.split('_')[-1][:-4])) #sort on frame number
        pathlist = [os.path.join(switch[mode], filename) for filename in filenamelist]
        return filenamelist, pathlist 
        
              
    def loadRowRangesDict(self):
        rowRanges = [i for i in xrange(0,108,12)]
        rowRanges_tuples = [(val, rowRanges[i+1]) for i, val in enumerate(rowRanges[:-1])]
        rowLabels  = list(map(chr, range(ord('A'), ord('H')+1)))
        rowRangesDict = dict(zip(rowLabels, rowRanges_tuples))
        return rowRangesDict
    
    def fetchImjProcPlates(self, screen = 'ROS'):
        #define directory path
        assaysdir = os.path.join(self.basedir, '%s\Assays' %screen) 
        plateIds = os.listdir(assaysdir)
        #fecth lists
        processedList = []
        unprocessedList = []
        for Id in plateIds:
            imgseqpath = os.path.join(assaysdir, '%s\ImgSeq' %Id)
            if os.path.exists(imgseqpath):
                processedList.append(Id)
            else:
                unprocessedList.append(Id)     
        return processedList, unprocessedList
    
    def fetchTrackedPlates(self, screen = 'ROS'):
        #fetch ImageJ processed plates
        processedList, unprocessedList = self.fetchImjProcPlates(screen = screen)
        #fetch all tracked plates
        trackdir = os.path.join(self.pickledir, 'PlateFlyTracks') 
        filelist = os.listdir(trackdir)
        trackedPlates = ['_'.join(filename.split('_')[:-1]) for filename in filelist]
        #filter out tracked plates for the specific screen
        untrackedList = [plateId for plateId in processedList if plateId not in trackedPlates]
        trackedList = [plateId for plateId in processedList if plateId in trackedPlates]   
        return trackedList, untrackedList
    
    def fetchRepeats(self):
        from itertools import chain
        from collections import Counter
        #define batch keys
        snames = ['ROS', 'Starvation']
        batchdates = self.batchDates()
        numbatches = [len(batchdates[sname]) for sname in snames]
        batchkeys = [['batch%s' %str(i+1) for i in xrange(dset)] for dset in numbatches]
        #fetch clustered plateIDs
        clusterPlateIDs = [self.loadClusterPlateIDs(screen = sname) for sname in snames]
        plateIDs = [[clusterPlateIDs[i][batch] for batch in dset] for i, dset in enumerate(batchkeys)]        
        plateIDs = [list(chain(*[list(chain(*batch.values())) for batch in dset])) for dset in plateIDs]#unpack
        #fetch repeats
        counter = [Counter([plateID.split('_')[0] for plateID in dset]) for dset in plateIDs]#count stocks in dataset
        repeats = [[stock for (stock, val) in dset.items() if val > 1 and stock.startswith('JS')] for dset in counter]
        repeats = dict(zip(snames, repeats))
        return repeats
    
    def buildU96RecordsObj(self):
        import cPickle as pickle
        import sqlite3
        from itertools import chain
        dbo = DatabaseOperations()
        #Connect to database
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        #fetch tablenames from database
        cursor.execute('''SELECT name FROM sqlite_sequence''')
        tablenames = cursor.fetchall(); tablenames = list(chain(*tablenames))#unpack
        db.close()
        #build dictionary object
        u96RecordsDict = {}
        for tablename in tablenames:
            #fetch records where sampleN = 96 from database
            records = dbo.fetchRecord(96, 'SampleN', tablename = tablename)
            records = [(record[-2], record[9]) for record in records]#parse records
            fwIds, sampleN = zip(*records)
            #fetch plateset
            plateset = dbo.fetchPlateSet()
            plateset = list(plateset[tablename])
            #select undersampled assays from plateset
            fwIds_u96 = [assay for assay in plateset if assay not in fwIds]
            #fetch records from database for undersampled assays
            records_u96 = [dbo.fetchRecord(assay, 'fwID', tablename = tablename) for assay in fwIds_u96]
            records_u96 = list(chain(*records_u96))#unpack
            records_u96 = [(record[-2], record[9]) for record in records_u96]#parse records
            u96RecordsDict[tablename] = records_u96
        #serialise dictionary
        picklepath = os.path.join(self.pickledir, 'u96Records.pickle')
        with open(picklepath, 'wb') as f:
            pickle.dump(u96RecordsDict, f, protocol = 2)
        return
    
    def loadU96Records(self):
        import cPickle as pickle
        picklepath = os.path.join(self.pickledir, 'u96Records.pickle')
        with open(picklepath, 'rb') as f:
            u96Records = pickle.load(f)
        return u96Records
    
    def buildWellCensorRegistry(self):
        from itertools import chain
        import cPickle as pickle    
        u96Records = self.loadU96Records()
        u96Records_unpacked = list(chain(*u96Records.values()))
        wellCensorRegistry = {}
        for (fwId, sampleN) in u96Records_unpacked:
            try:
                censoredWells = PlateTracker(fwId).detectEmptyWells()
                wellCensorRegistry[fwId] = censoredWells
                if not 96-len(censoredWells) == sampleN:
                    print('Plate %s: number of empty wells detected (%s) does not match record of number of wells loaded (%s)' %(fwId, len(censoredWells), sampleN))
            except ValueError, e:
                print(e)
                continue
        #serialise dictionary
        picklepath = os.path.join(self.pickledir, 'wellCensorRegistry.pickle')
        with open(picklepath, 'wb') as f:
            pickle.dump(wellCensorRegistry, f, protocol = 2)
        return

    def loadWellCensorRegistry(self):
        import cPickle as pickle
        #load dictionary
        picklepath = os.path.join(self.pickledir, 'wellCensorRegistry.pickle')
        with open(picklepath, 'rb') as f:
            wellCensorRegistry = pickle.load(f)
        return wellCensorRegistry
        
    def writeCensorRegistryToFile(self):
        #load data object
        censorRegistry = self.loadWellCensorRegistry()
        #headinggs
        headings = '%s\t%s\t%s\n' %('fwID', 'censoredWells', 'SampleN')
        lines = []; lines.append(headings)
        #parse lines
        for key in censorRegistry.keys():
            censoredWells = censorRegistry[key]
            sampleN = 96-len(censoredWells)
            censoredWells_string = ','.join(censoredWells)
            line = '%s\t%s\t%s\n' %(key, censoredWells_string, sampleN)
            lines.append(line)
        #write lines to file
        filepath = os.path.join(self.workdir, 'wellCensorRegistry.txt')
        with open(filepath, 'w') as f:
            f.writelines(lines)
        return 
                                                                                                                                                                                                                                                                                                                                                                                                

class PlateTracker(FwObjects):
    
    def __init__(self, fwId, custompath = ''):       
        FwObjects.__init__(self)
        self.fwId = fwId
        self.timelabel = ''
        self.custompath = os.path.join(custompath, self.fwId)
        #make stockdir if it does not exists
        if isinstance(self.fwId, (str, unicode)):
            stockdir = os.path.join(self.workdir, self.fwId)
            if os.path.exists(stockdir):
                self.stockdir = stockdir
            else:
                os.makedirs(stockdir)
                self.stockdir = stockdir
        else:
            self.stockdir = self.workdir
        return
        
    def getMembers(self):
        import inspect
        methods = inspect.getmembers(self, predicate = inspect.ismethod)
        print(methods)
        return
        
    def copyImgSeqToContours(self, output = 'remote'):
        import shutil
        filelist, pathlist = self.loadPlateImgSeq()
        #set contours directory
        if output == 'remote':
            imgseqdir = os.path.split(os.path.split(pathlist[0])[0])[0]
            contoursdir = os.path.join(imgseqdir, 'Contours')
        elif output == 'local':
            contoursdir = os.path.join(self.stockdir, 'Contours')
        #test whether contoursdir exists and delete content
        if os.path.exists(contoursdir):
            if len(os.listdir(contoursdir)) > 0:
                [os.remove(os.path.join(contoursdir, filename)) for filename in os.listdir(contoursdir)]
        else:
            os.makedirs(contoursdir)
        #copy and rename files 
        for i, path in enumerate(pathlist):
            destpath = os.path.join(contoursdir, 'frame_%03d.jpg' %(i+1))
            shutil.copyfile(path, destpath)
        return contoursdir
    
    def renameImageSeq(self, screen = 'ROS', output = 'remote'):
        #set contours directory
        if output == 'remote':
            contoursdir = os.path.join(self.basedir, '%s\Assays\%s' %(screen, self.fwId), 'Contours')
        elif output == 'local':
            contoursdir = os.path.join(self.stockdir, 'Contours')
        elif output == 'custom':
            contoursdir = os.path.join(self.custompath, 'Contours')
        #rename files
        filelist = [filename for filename in os.listdir(contoursdir) if filename.startswith('FW')]
        assert (len(filelist)>0), 'The directory path %s contains no image files.' %contoursdir
        filelist = sorted(filelist, key = lambda x: int(x.split('-')[2] + x.split('-')[3][:-4]))
        for i, f in enumerate(filelist):
            filepath = os.path.join(contoursdir, f)
            new_filepath = os.path.join(contoursdir, 'frame_%03d.tif' %(i+1))
            os.rename(filepath, new_filepath)
        return
    
    def fetchWellCoordinates(self, frame, centroid):
        cX, cY = centroid
        (h,w) = frame.shape[:2]
        roi_h = list(np.linspace(0, h, 9))
        roi_w =list( np.linspace(0, w, 13))
        roi_h.append(cY); roi_w.append(cX)
        roi_h.sort(); roi_w.sort()
        rowIdx = roi_h.index(cY); colIdx = roi_w.index(cX)
        wellIdx = (rowIdx-1) * 12 + colIdx
        return wellIdx
    
    def plateFlytracker(self, output = 'remote'):
        import cv2
        from collections import OrderedDict
        import cPickle as pickle
        #copy image sequence to contoursdir and rename images
        print('Copying %s image sequence to contours dir. \n' %self.fwId)
        contoursdir = self.copyImgSeqToContours(output = output)
        #set variables
        counter = 0
        plateFlytracks = OrderedDict()
        #inititate videocapture and build background subtractor object
        filepath = os.path.join(contoursdir, 'frame_001.jpg')    
        cap = cv2.VideoCapture(filepath)
        fgbg = cv2.createBackgroundSubtractorKNN(history = 10, detectShadows = False)
        #apply brackground subtractor
        while(1):
            counter +=1
            ret, frame = cap.read()
            #test whether frame is empty 
            if np.count_nonzero(frame) == 0:
                break
            #apply background subtractor and filter out small contours
            framename = 'frame_%03d' %counter    
            print('Processing %s' %framename)    
            cv2.GaussianBlur(frame, (5,5), 0)
            fgmask = fgbg.apply(frame)
            im2, cnts, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 1]
            #cluster contours for each well
            wellsMapOfContours = [[] for i in xrange(96)]
            for i, cnt in enumerate(cnts):
                #compute the center of the contour
                M = cv2.moments(cnt) 
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid = (cX, cY)
                #compute the mean intensity of the contour
                mask = np.zeros(frame.shape, np.uint8)
                cv2.drawContours(mask, cnts, i, 255, -1)
                mean = cv2.mean(frame[mask == 255])
                #map contour to well
                wellIdx = self.fetchWellCoordinates(frame, centroid)
                wellsMapOfContours[wellIdx-1].append((self.wellIds[wellIdx-1], cnt, i, mean, centroid))
            #sort and filter contours in each well on the basis of their mean intensity and area
            wellsMapOfContours_sorted = [sorted(well, key = lambda x: x[3]) for well in wellsMapOfContours]
            wellsMapOfContours_filtered = [well[0] if len(well)>0 else well for well in wellsMapOfContours_sorted]
            wellsMapOfContours_filtered = [well if len(well)>0 and 100 >= cv2.contourArea(well[1]) > 10 else [] for well in wellsMapOfContours_filtered]
            binTracker = [(self.wellIds[i], counter, 1) if len(well)>0 else (self.wellIds[i], counter, 0) for i, well in enumerate(wellsMapOfContours_filtered)]
            #filter out contours
            contourIdx = [well[2] for well in wellsMapOfContours_filtered if len(well)>0]
            frameContours = [cnt for i, cnt in enumerate(cnts) if i in contourIdx]
            plateFlytracks[framename] = [frameContours, wellsMapOfContours_filtered, binTracker]
            #draw contours on frame and save image
            cv2.drawContours(frame, frameContours, -1, (255,255,0), 1)
            if output == 'remote':
                contourspath = os.path.join(contoursdir, '%s.jpg' %framename)
            elif output == 'local':
                workdir_stock = os.path.join(self.workdir, self.fwId)
                contoursdir = os.path.join(workdir_stock, 'Contours')
                contourspath = os.path.join(contoursdir, '%s.jpg' %framename)  
            cv2.imwrite(contourspath, frame)
        #serialise plateFlytracks
        picklepath = os.path.join(self.tracksdir, '%s_flytracks.pickle' %self.fwId)
        with open(picklepath, 'wb') as f:
            pickle.dump(plateFlytracks, f, protocol = 2)
        #update flyTracks dictionary
        trackdataMap = self.loadTrackdataPlatesMap()
        trackdataMap[self.fwId] = picklepath
        picklefile = os.path.join(self.pickledir, 'trackdataPlatesMap.pickle')
        with open(picklefile, 'wb') as f:
            pickle.dump(trackdataMap, f, protocol = 2)
        return 
    
    def calculateToD_timestamp(self, fnumberTod):
        import datetime
        filelist, pathlist = self.loadPlateImgSeq() 
        frame1 = os.path.splitext(filelist[0])[0]
        #Find time zero 
        flywheel, slot, date, time = frame1.split('-')
        startDate = '%s-%s-%s' %(date[:4], date[4:6], date[6:])
        startTime = '%s:%s:%s' %(time[:2], time[2:4], time[4:])
        start_timestamp_str = '%s %s' %(startDate, startTime)
        startDatetime = datetime.datetime.strptime(start_timestamp_str, "%Y-%m-%d %H:%M:%S")
        #Generate datetime objects from tod framenumbers
        todDatetimes= []
        for well in fnumberTod:
            wellID, framenumber_tod, censorship = well
            filename_tod = filelist[framenumber_tod-1]
            flywheel, slot, date, time = os.path.splitext(filename_tod)[0].split('-')
            date = '%s-%s-%s' %(date[:4], date[4:6], date[6:])
            time = '%s:%s:%s' %(time[:2], time[2:4], time[4:])
            timestamp_str = '%s %s' %(date, time)
            todDatetime = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            todDatetimes.append(todDatetime)
        #Calculate datetime differences (todDatetime - startDatetime)    
        diff_datetimes = [tod - startDatetime for tod in todDatetimes]
        todHours = [(datetime.days * 24 + datetime.seconds/3600.0) for datetime in diff_datetimes]
        plateTod = [(fnumberTod[i][0], fnumberTod[i][1], fnumberTod[i][2], tod) for i, tod in enumerate(todHours)]
        #Sort and update ToDtuples and calculate survival percentages 
        plateTod = sorted(plateTod, key = lambda x: x[3])
        size = len(todDatetimes)
        survPercentage = [((size-i-1)/float(size)*100) for i in xrange(size)]
        for i, item in enumerate(plateTod):
            welltuple = list(item)
            welltuple.append(survPercentage[i])
            welltuple = tuple(welltuple)
            plateTod[i] = welltuple
        #insert time 0   
        plateTod.insert(0,  ('t0', 0, 1, 0, 100.0))
        return plateTod
    
    def calculateToD_framenumber(self, fnumberTod, period = 20):
        #calculate time of death (tod) from tod_framenumbers
        todTimes= []
        for well in fnumberTod:
            wellID, framenumber_tod, censorship = well
            tod_minutes = framenumber_tod * period
            tod_hours = tod_minutes/60.0
            todTimes.append(tod_hours)       
        plateTod = [(fnumberTod[i][0], fnumberTod[i][1], fnumberTod[i][2], tod) for i, tod in enumerate(todTimes)]
        #Sort and update ToDtuples and calculate survival percentages 
        plateTod = sorted(plateTod, key = lambda x: x[3])
        size = len(todTimes)
        survPercentage = [((size-i-1)/float(size)*100) for i in xrange(size)]
        for i, item in enumerate(plateTod):
            welltuple = list(item)
            welltuple.append(survPercentage[i])
            welltuple = tuple(welltuple)
            plateTod[i] = welltuple
        #insert time 0   
        plateTod.insert(0,  ('t0', 0, 1, 0, 100.0))
        return plateTod      
                 
   
    def plateToDCrawler(self, period = 20):
        from itertools import compress
        import cPickle as pickle
        #load data objects
        plateFlytracks = self.loadPlateFlytracks()
        trackdata = plateFlytracks.values()
        censorRegistry = self.loadWellCensorRegistry()
        try:
            censoredWells = censorRegistry[self.fwId]
        except KeyError:
            censoredWells = []
        #unpack data
        plate_bintracker = [frame[-1] for frame in trackdata]
        plate_bintracker = zip(*plate_bintracker)
        plate_bintracker = [zip(*well) for well in plate_bintracker]
        #scroll wells and find best tod estimate
        fnumberTod = []
        for well in plate_bintracker:
            wellID, framenumbers, bintracker = well
            print('Estimating ToD in %s well.' %wellID[0])
            #filter out frames where no movement was detected
            fnumbers_filtered = list(compress(framenumbers, bintracker))
            #test whether at least one estimate was found
            try:
                assert (wellID[0] not in censoredWells)
                assert (len(fnumbers_filtered) > 0)
            except AssertionError:
                 estimateTod = (wellID[0], 1, 0)
                 fnumberTod.append(estimateTod)
                 continue
            #filter out tracking discontinuities that are shorter than 5 frames 
            #define variables 
            size = len(fnumbers_filtered)
            lastcnt = fnumbers_filtered[-1]
            if size > 1:
                todFrame_estimates = [(i, numb, (fnumbers_filtered[i+1] - numb)) for i, numb in enumerate(fnumbers_filtered[:-1]) if (fnumbers_filtered[i+1]-numb) >= 5]
                todFrame_estimates.append((size-1, lastcnt, framenumbers[-1]-lastcnt))#append last estimate
            elif size == 1:
                todFrame_estimates = [(0, fnumbers_filtered[0], framenumbers[-1]- fnumbers_filtered[0])]#a single contour detected
            
            #if only one candidate estimate was found test whether it is the last frame
            if len(todFrame_estimates) == 1:
                (idx, framenumb, dist) = todFrame_estimates[0]
                if dist >= 3:
                    estimateTod = (wellID[0], framenumb+1, 1)
                else:
                    estimateTod = (wellID[0], framenumbers[-1], 0)
                fnumberTod.append(estimateTod)
                continue
            #if not sweep through the estimates
            elif len(todFrame_estimates) > 1:        
                for i , estimate in enumerate(todFrame_estimates):
                    (idx, framenumb, dist) = estimate
                    
                    #test whether estimate[i] is the last element of the list
                    if i+1 == len(todFrame_estimates):
                        if framenumb == framenumbers[-1]:
                            estimateTod = (wellID[0], framenumbers[-1], 0)
                        else:
                            if dist >=3:
                                if (framenumb + dist) - fnumbers_filtered[-1] < 2:
                                    estimateTod = (wellID[0], framenumbers[-1], 0)
                                else:
                                    estimateTod = (wellID[0], framenumb+1, 1)
                            else:
                                estimateTod = (wellID[0], framenumbers[-1], 0)
                        fnumberTod.append(estimateTod)
                        break
                    #if not, filter estimate[i] on the size of the tracking gap to estimate[i+1]
                    nextmove_frame = framenumb + dist#define next move
                    nextstop_frame = todFrame_estimates[i+1][1]#define next stop
                    if 5 <= dist <= 30:
                        if nextstop_frame - nextmove_frame >= 3:#there is movement between next move and next stop
                            continue
                        else:
                            #estimate whether the short tracking sequence detected - no more than 2 frames - is a false positive
                            #fetch list of estimates after estimate[i] where the same happens 
                            estimateSubset = todFrame_estimates[i:-1]
                            falsePositives_selector = [1 if frame+dist == todFrame_estimates[i+1+j][1] else 0 for j, (idx, frame, dist) in enumerate(todFrame_estimates[i:-1])]
                            falsePositives= list(compress(estimateSubset, falsePositives_selector))
                            #test whether false positives exist and whether estimate[i] is included
                            #all but one estimate must be a false positive
                            try:
                                assert (len(falsePositives) > 0 and falsePositives[0] == estimate)
                                assert(len(estimateSubset) - len(falsePositives) <=1)
                            except AssertionError:
                                continue
                            #recalculate tracking gap after eliminating false positives
                            (idx_fp, frame_fp, dist_fp)= lastFalsePositive = falsePositives[-1]
                            nextmove_fp = (frame_fp + dist_fp) - framenumb
                            #tracking gaps bigger than 30 frames
                            if nextmove_fp >= 30:
                                estimateTod = (wellID[0], framenumb+1, 1)
                                fnumberTod.append(estimateTod)
                                break
                            else:
                                continue 
                    #no tracking for more than 30 frames
                    elif dist > 30:
                        estimateTod = (wellID[0], framenumb+1, 1)
                        fnumberTod.append(estimateTod)
                        break          
        #update plateTod censorship regarding deaths in the first 2 hours
        fnumberTod = [(wellID, fnumber, censor) if fnumber >=6 else (wellID, fnumber, 0) for (wellID, fnumber, censor) in fnumberTod]
        #calculate time of deaths for each weel from frame numbers
        if self.timelabel == 'timestamp': 
            plateTod = self.calculateToD_timestamp(fnumberTod)
        elif self.timelabel == 'framenumber':
            plateTod = self.calculateToD_framenumber(fnumberTod, period = period)
        #serialize stockdata
        stockdata = [plateFlytracks, plateTod]
        stockdatapath = os.path.join(self.stockdatadir, '%s.pickle' %self.fwId)
        with open(stockdatapath, 'wb') as f:
            pickle.dump(stockdata, f, protocol = 2)
        #add path to stockdata to plateDatalists dictionary   
        self.plateDatalists[self.fwId] = stockdatapath
        with open(self.datalistspath, 'wb') as f:
            pickle.dump(self.plateDatalists, f, protocol = 2)
        #estimate KM median survival
        print('Estimating KM median survival. \n')
        medsurv, error = self.km_medSurv(stockdata)
        #add KM estimate to medSurv dictionary
        self.medSurv_dict[self.fwId] = [medsurv, error]
        medsurvpath = os.path.join(self.pickledir, 'medSurv_dict.pickle')
        with open(medsurvpath, 'wb') as f:
            pickle.dump(self.medSurv_dict, f, protocol = 2)        
        return stockdata
        

    def km_medSurv(self, stockdata, alpha = 0.01):
        from lifelines import KaplanMeierFitter as kmf
        #unpack data
        try:
            [plateFlyTracks, todplate] = stockdata
        except ValueError:
            [plateFlyTracks, todplate] = stockdata[0]  
        #Estimate km median survival
        kmf = kmf()
        kmf.fit(zip(*todplate)[3], event_observed = zip(*todplate)[2])
        medsurv = kmf.median_
        #fetch medsurv error: survival
        survErr = list(kmf.confidence_interval_.loc[kmf.median_].values)
        #fetch medsurv error: time
        ci_data = zip(kmf.confidence_interval_.index, kmf.confidence_interval_.values)
        while alpha:
            timeErr_seq = [tupl[0] for tupl in ci_data if abs(tupl[1][0]-0.5) < alpha or abs(tupl[1][1]-0.5) < alpha]
            if len(timeErr_seq) >= 2:
                timeErr_seq = [[val for val in timeErr_seq if val - medsurv <= 0], [val for val in timeErr_seq if val - medsurv > 0]]
                try:
                    lowerbound = max(timeErr_seq[0])
                    upperbound = min(timeErr_seq[1])
                    timeErr = [lowerbound, upperbound]
                    error = [timeErr, survErr]
                    break
                except ValueError:
                    pass                    
            alpha += 0.01
        return medsurv, error
                                     
    
    def plateAnalyser(self, period = 20, output = 'remote'):
        print('Analysing plate %s\n' %self.fwId)
        self.plateFlytracker(output = output)
        stockdata =  self.plateToDCrawler(period = period)
        return stockdata
    
    def detectEmptyWells(self, showmax = False):
        from Flywheel_AlignFunctions import Maxima
        from itertools import ifilter, compress
        from math import ceil
        import cv2
        dbo = DatabaseOperations()
        #fetch pathlist for plate image sequence
        filelist, imgpathlist = self.loadPlateImgSeq()
        #estimate threshold
        images = [cv2.imread(imgpath, 0) for imgpath in imgpathlist[:11]]
        histograms = [np.histogram(image.ravel(),256,[0,256]) for image in images]
        mean = np.mean([np.sum(hist[:50]*bins[:50])/50.0 for hist, bins in histograms])
        threshold = ceil((50*(721.356363636/mean)))
        #detect maxima
        maxCoord = []; imgMat = [] 
        for imagepath in imgpathlist[:11]:
            image_resize = cv2.imread(imagepath, 0)
            coordinates = Maxima().fetchMaxCoordinates(image_resize, xlim = 600, ylim = 400)
            maxCoord.append(coordinates); imgMat.append(image_resize)
        #filter out maxima if int>threshold
        counter = 0; tempList = []  
        while(1):     
            maxima_idx = [[j for j, coordinate in enumerate(imgCoord) if imgMat[i][coordinate] <= threshold] for i, imgCoord in enumerate(maxCoord)]#threshold filter
            maxCoord_filtered = [maxCoord[i][val] for i, img in enumerate(maxima_idx) for val in img]#select maxima based on threshold filtering
            #fetch wells coordinates from masterplate    
            wellsmap = self.xywellMap(); wellIDs = wellsmap.keys()
            #map maxima to wells 
            maxWellsmap = []
            for Id in wellIDs:
                xwell, ywell = wellsmap[Id]
                maxCluster = list(ifilter(lambda x: np.sqrt(abs(x[1]-xwell)**2 + abs(x[0]-ywell)**2)<=21, maxCoord_filtered))#filter out maxima outside wells
                maxWellsmap.append((Id, maxCluster))
            #fetch list of censored wells
            censoredWells = [well[0] for well in maxWellsmap if len(well[1])==0]; estimateSampleN = 96-len(censoredWells)
            tempList.append((dbo.fetchRecSampleN(self.fwId)-estimateSampleN, censoredWells))
            print(dbo.fetchRecSampleN(self.fwId)-estimateSampleN, threshold)
            #test whether number of censored wells matches record of empty wells
            if estimateSampleN == dbo.fetchRecSampleN(self.fwId):
                break
            #test whether treshold sequence does not lead to convergence 
            elif counter>=3:
                booleanList = [0 if val<0 else 1 for val in zip(*tempList)[0]]
                tempList_filtered = list(compress(tempList, booleanList))
                try:
                    assert len(tempList_filtered)>0
                    if len(tempList_filtered) < len(tempList):
                        censoredWells = [tempList_filtered[0][1] if booleanList[0]==0 else tempList_filtered[-1][1]][0]
                        break
                except AssertionError:
                    pass
            threshold = [threshold + 1 if estimateSampleN < dbo.fetchRecSampleN(self.fwId) else threshold - 1][0]#if not re-adjust threshold
            counter +=1
        #sort censored wells list  
        censoredWells = sorted(censoredWells, key = lambda x:(x[0], int(x[1:])), reverse = True)
        #display detected maxima on plate image
        if showmax:
            for well in maxWellsmap:
                if len(well[1])>0:
                    image_resize = cv2.imread(imgpathlist[0], 0)
                    [cv2.circle(image_resize, (int(y),int(x)), 3, (255,255,255), 1) for (x,y) in well[1]]
                    cv2.imshow('maxima', image_resize)
                    cv2.waitKey(0)
        return censoredWells

    
                
class DataVis(FwObjects):
    
    def __init__(self, fwId, custompath = ''):
        FwObjects.__init__(self)
        self.fwId = fwId
        self.timelabel = ''
        #make stockdir if it does not exists
        if isinstance(self.fwId, (str, unicode)):
            stockdir = os.path.join(self.workdir, self.fwId)
            self.custompath = os.path.join(custompath, self.fwId)
            if os.path.exists(stockdir):
                self.stockdir = stockdir
            else:
                os.makedirs(stockdir)
                self.stockdir = stockdir
        else:
            self.stockdir = self.workdir
            self.custompath = self.workdir
        return
       
    def getMembers(self):
        import inspect
        methods = inspect.getmembers(self, predicate = inspect.ismethod)
        print(methods)
        return   
        
    def fetchStockdata(self):
        import cPickle as pickle
        #test datatype
        if isinstance(self.fwId, (str, unicode)):
            IDs = [self.fwId]
        elif isinstance(self.fwId, (tuple, list)):
            IDs = self.fwId
        else:
            raise Exception('Datatype error: plates IDs must be either str or sequence (tuple, list) type. \n')
        #test whether all IDs are valid fwIds
        matches = [Id for Id in IDs if Id in self.plateDatalists.keys()]
        mismatches = [Id for Id in IDs if Id not in self.plateDatalists.keys()]
        #fetch stockdata 
        stockdata = []
        if len(matches) > 0:
            if len(matches) == 1:
                self.fwId = matches[0]
            else:
                self.fwId = matches                
            stockdatapath = [self.plateDatalists[key] for key in matches]
            for path in stockdatapath:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    stockdata.append(data)
        #test whether at least some mismatches are either plateIDs or stockIDs in the database
        if len(mismatches)> 0:
            #test whether mismatches are plateIDs in the database not yet analysed
            tablePlatesets = DatabaseOperations().fetchPlateSet()
            dbPlateset = set([plate for table in tablePlatesets.values() for plate in table])
            plateID_mismatches = [mismatch for mismatch in set(mismatches) if mismatch in dbPlateset]
            if len(plateID_mismatches) > 0:
                print('%s is/are in the database but has/have yet to be analysed. \n' %(','.join(plateID_mismatches)))
            mismatches = list(set(mismatches).difference(plateID_mismatches))
            #test whether mismatches are stockIDs
            if len(mismatches) > 0:
                tableStocksets = DatabaseOperations().fetchStockset()
                dbStockset = set([stock for table in tableStocksets.values() for stock in table])
                stockIDs = [mismatch for mismatch in set(mismatches) if mismatch in dbStockset]
                stockID_mismatches = list(set(mismatches).difference(stockIDs))
                if len(stockIDs) > 0:
                    if len(stockIDs) == 1:
                        print('%s was excluded because it is not a valid plateID. However, it is a valid stockID. \n' %(','.join(stockIDs)))
                    else:
                        print('%s were excluded because they are not valid plateIDs. However, they are valid stockIDs. \n' %(','.join(stockIDs)))
                    plateIDs = [DatabaseOperations().fetchPlateIDFromStock(stockID, tablenames = ['ROS', 'Starvation']) for stockID in stockIDs]
                    for i, stockID in enumerate(stockIDs):
                        print('PlateIDs in the database for %s: %s\n' %(stockID, plateIDs[i]))        
                if len(stockID_mismatches) > 0:
                    print('%s is/are not in the database. \n' %(','.join(stockID_mismatches)))
        #assert whether at least one match was found
        assert (len(matches) > 0), '%s has no matches in the database' %self.fwId
        return stockdata
        
        
    def fetchMedianSurvData(self, screen , stockID):
        from itertools import chain
        #fetch flywheel IDs
        clusterPlateIDs = self.loadClusterPlateIDs(screen = screen)#load clusterPlateIDs object
        dsetbatches = len(clusterPlateIDs.keys())#number of batches in dataset
        clusterPlateIDs = [clusterPlateIDs['batch%s' %str(val+1)] for val in xrange(dsetbatches)]#fetch plates in each batch
        clusterPlateIDs = [list(chain(*batch.values())) for batch in clusterPlateIDs]#flatten wheels in each batch 
        fwIds = [[fwId for fwId in batch if fwId.split('_')[0]==(stockID)] for batch in clusterPlateIDs]
        numbatches = [i+1 for i, batch in enumerate(fwIds) if len(batch)>0]
        #fetch data
        medsurvDict = self.loadMedSurv()#load data object
        data = [[medsurvDict[fwId] for fwId in batch if fwId.split('_')[-1]!='299'] for batch in fwIds]
        data = [zip(*batch) for batch in data if len(batch)>0]
        data = [(med, [assay[0] for assay in err]) for med, err in data]
        medsurvdata = [medians, errors] = zip(*data)
        return medsurvdata, numbatches 
           
            
    def writePlateToD(self):
        #load stockdata and unpack
        stockdata = self.loadStockdata()
        [plateFlyTracks, todplate] = stockdata
        #Write ToD.txt file
        todfile = '%s_plateToD.txt' %self.fwId
        todpath = os.path.join(self.stockdir, todfile) 
        with open(todpath, 'w') as f:
            heads = ['well_Id', 'frameN', 'Censored', 'tod', 'survPerc']
            headline = '\t'.join(heads) + '\n'
            f.write(headline)
            for item in todplate:
                line = '%s\t%i\t%i\t%.2f\t%.2f\n' %(item[0], item[1], item[2], item[3], item[4])
                f.write(line)     
        return     
                  
                
    def platePlotter(self, todlabel = 'frame'):
        from matplotlib import pyplot as plt
        #fetch stockdata
        try:
            stockdata = self.fetchStockdata()
            if len(stockdata)==0:
                sys.exit()
        except AssertionError, e:
            print(e)
            sys.exit()
        #unpack data
        try:
            [plateFlyTracks, todplate] = stockdata    
        except ValueError:
            [plateFlyTracks, todplate] = stockdata[0]
            
        plt.figure('Plate %s' %self.fwId)
        ax = plt.subplot(111)
        #define x and y sets
        xset = [[i+1]*8 for i in xrange(12)] 
        xset = zip(*xset)
        xset = [val for sublist in xset for val in sublist]
        yset = [[i+1]*12 for i in xrange(8)]
        yset = [val for sublist in yset for val in sublist]
        #define colorMap
        todTimes_dict = dict(zip([tupl[0] for tupl in todplate], todplate))
        rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        well_labels = [['%s%i' %(row, i+1) for i in xrange(12)] for row in rows[::-1]]
        well_labels = [label for row in well_labels for label in row][::-1]#unpack  
        #fetch tod times
        todTimes = []
        for label in well_labels:
            try:
                tod = todTimes_dict[label][3]   
            except KeyError:
                tod == 0
            todTimes.append(tod)
        #variables definition
        maxval = max(todTimes)
        saturation = [val/float(maxval) for val in todTimes]
        colorMap = [(s,1-s,s) for s in saturation]
        #define plot
        ax.scatter(xset, yset, s = 700, c = colorMap, edgecolor = 'None')
        #text annotations definitions
        if todlabel == 'frame':
            [ax.annotate(int(todplate[i+1][1]), (xset[i], yset[i]), fontsize = 8, horizontalalignment='center', verticalalignment='center') for i in xrange(len(todTimes))]
        elif todlabel == 'time':
            [ax.annotate(int(todTimes[i]), (xset[i], yset[i]), fontsize = 8, horizontalalignment='center', verticalalignment='center') for i in xrange(len(todTimes))]  
        #define ticks, labels and axis limits
        ax.set_xlim([0, 13])
        ax.set_xticks(xset)
        ax.set_xticklabels(xset[::-1])
        ax.set_ylim([0,9])
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_yticklabels(rows)
        plt.show()
        return
     
                                                                                                         
    def survPlotter(self, ci = True):
        from matplotlib import pyplot as plt
        from pylab import getp, setp
        from lifelines import KaplanMeierFitter as kmf
        #Fetch data
        try:
            stockdata = self.fetchStockdata()
            if len(stockdata)==0:
                sys.exit()
        except AssertionError, e:
            print(e)
            sys.exit()
        #Set data
        if isinstance(self.fwId, str):
            self.fwId = [self.fwId]
            labels = ['KM estimate']
        else:    
            #Labels
            labels = ['%s_%s' %(Id.split('_')[0], Id.split('_')[2]) for Id in self.fwId] #labels = ['%s' %Id for Id in self.fwId]
        #Plot survival curve
        kmf = kmf()
        censorArrs = [zip(*stock[1])[2] for stock in stockdata]
        todArrs = [zip(*stock[1])[3] for stock in stockdata]
        for i, arr in enumerate(todArrs):
            kmf.fit(arr, event_observed = censorArrs[i], label = labels[i])
            if i == 0:
                ax1 = kmf.plot(ci_show = ci)
                if len(self.fwId) > 1:
                    c = getp(ax1.lines[i], 'color')
                else:
                    c = 'r'
                ax1.plot([kmf.median_,kmf.median_],[0,0.5], color = c, linestyle = '--', alpha = 0.5)
                ax1.plot([0,kmf.median_],[0.5,0.5], color = c, linestyle = '--', alpha = 0.5)
            else:
                ax1 = kmf.plot(ax = ax1, ci_show = ci)
                c = getp(ax1.lines[-1], 'color')
                ax1.plot([kmf.median_,kmf.median_],[0,0.5], color = c, linestyle = '--', alpha = 0.5)
                ax1.plot([0,kmf.median_],[0.5,0.5], color = c, linestyle = '--', alpha = 0.5)
        #title, labels and annotations
        if len(self.fwId) == 1:
            title = '%s' %self.fwId[0] #'%s_%s' %(self.fwId[0].split('_')[0], self.fwId[0].split('_')[2]) 
            plt.title('%s KaplanMeier fit' %title)
            medianSurv = '%.1f' %kmf.median_
            ax1.annotate(medianSurv, (kmf.median_,0.5), fontsize = 10)
        else:
            plt.title('KaplanMeier fit')   
        ax1.set_xlabel('Time (hours)', fontsize = 12)
        ax1.set_ylabel('Survival probability', fontsize = 12)
        leglabels = ax1.get_legend().get_texts()
        [setp(label, fontsize = 11) for label in leglabels]
        plt.show()
        return
      
    
    def stockMedSurv(self, screen , stockID, plottype = 'barplot'):
        from itertools import chain                                                                                                                                                                        
        import matplotlib.pyplot as plt
        import numpy as np
        #fetch medsurvdata for test stock
        medsurvdata, numbatches = self.fetchMedianSurvData(screen, stockID)
        medians_test, errors = medsurvdata
        if plottype == 'barplot':
            #set bar plot
            ax = plt.subplot(111)
            batchsize = len(medians_test)
            xset = np.arange(1,(batchsize*2)+2,2)
            for i, batch in enumerate(medians_test):
                start = xset[i] - (0.5*len(batch))/2.0
                end = start + (0.5*(len(batch)-1))
                xarr = np.linspace(start,end, len(batch))
                err = [abs(np.asarray(assay)-batch[j]) for j, assay in enumerate(errors[i])]
                err = zip(*err)
                ax.bar(xarr, batch, width = 0.5, color = 'b', alpha = 0.2, yerr = err)
            #set ticks and labels     
            ax.set_xticks(xset)
            xlabels = ['batch%s'%val for val in numbatches]
            ax.set_xticklabels(xlabels, rotation = 60)
            ax.set_ylabel('Median survival (hours)', fontsize = 13)
        elif plottype == 'swarmplot':
            import seaborn as sns
            #fetch medsurvdata for empty
            medsurvdata_emp, _ = self.fetchMedianSurvData(screen, 'Empty')
            medians_emp, errors_emp = medsurvdata_emp
            #unpack medsurvdata
            medians_test = list(chain(*medians_test))
            medians_emp = list(chain(*medians_emp))
            #define data to plot
            medians = [medians_emp, medians_test]
            yset = medians_emp + medians_test 
            xset = list(chain(*[[i+1]*len(medians[i]) for i in xrange(2)]))
            hue = list(chain(*[[val]*len(medians[i]) for i, val in enumerate([1,2])]))
            #define plot
            fig, ax = plt.subplots(1,1)
            sns.swarmplot(x = xset, y = yset, s = 5, hue = hue, ax = ax, zorder = -1)
            #plot means
            means = [np.mean(sublist) for sublist in medians]
            ax.scatter([0,1], means, marker = '_', s = 200, color = 'r', linewidth = 2.0)
            #set xtick labels
            ax.set_xticklabels(['Empty', stockID], fontsize = 11, rotation = 60)
            #set ylabel
            ax.set_ylabel('Median survival (hours)', fontsize = 12)
            ax.legend('')
        plt.show()
        return
            
    
    def plotDataset(self, tablename, showstock = 'Empty', showassay = False):
        import numpy as np
        from scipy.stats import sem
        from matplotlib import pyplot as plt
        from matplotlib import lines
        from mpldatacursor import datacursor
        #define batch keys
        batchdates = self.batchDates()
        numbatches = len(batchdates[tablename])
        batchkeys = ['batch%s' %str(i+1) for i in xrange(numbatches)]
        #fetch clustered plateIDs
        clusterPlateIDs = self.loadClusterPlateIDs(screen = tablename)
        plateIDs = [clusterPlateIDs[batch] for batch in batchkeys]
        #filter out plateIDs not yet analysed
        plateset = DatabaseOperations().fetchPlateSet()
        plateset_table = plateset[tablename]
        stockdataIDs = [os.path.splitext(name)[0] for name in os.listdir(self.stockdatadir)]
        mismatches = [Id for Id in plateset_table if Id not in stockdataIDs]
        #fetch kmf_median and error for the remaining        
        data = [[[(Id, self.medSurv_dict[Id]) for Id in plate[i+1] if Id not in mismatches] for i in xrange(3)] for plate in plateIDs]
        #plot data
        fig = plt.figure()
        ax = plt.subplot(111)
        #colors and wheel labels definitions
        colors = ['b', 'r', 'g']
        labels = ['wheel1', 'wheel2', 'wheel3']
        #unpack batch data
        xticks_coord = []
        for i, batch in enumerate(data):
            for j, wheel in enumerate(batch):
                #unpack data
                try:
                    plateIDs, medArr = zip(*wheel)
                    medians, errors = zip(*medArr)
                    tErr, survErr = zip(*errors) 
                    #calculate wheel mean
                    wheel_mean = np.mean(medians)
                    errMean = sem(medians)
                    #set marker sizes and colors for each wheel and showstock
                    if showassay:
                        highlighter_idx = [z for z, plateId in enumerate(plateIDs) if plateId in showassay]
                    if showstock:
                        highlighter_idx = [z for z, plateId in enumerate(plateIDs) if showstock == plateId.split('_')[0]]
                    wheel_colors = [colors[j]]*len(medians)
                    sizes = [20]*len(medians)
                    if len(highlighter_idx) > 0:
                        for idx in highlighter_idx:
                            wheel_colors[idx] = '#AB30F2'
                            sizes[idx] = 50
                    #set xdata for each wheel
                    xset = [((i+0.75)+ 0.25*j)+i]*len(medians)        
                    wheelDset = ax.scatter(xset, medians, c = wheel_colors, s = sizes, alpha = 0.65)
                    ax.scatter(xset[0], wheel_mean, marker = '_', linewidth = 2.0, c = 'r', s = 50)#plot wheel means
                    #Label ax datapoints interactively
                    stocklabels = [Id.split('_')[0] for Id in plateIDs]
                    datacursor(wheelDset, hover=True, point_labels = stocklabels, fontsize = 10, bbox= None, xytext=(0, 25), formatter=lambda **kwargs: kwargs['point_label'][0])   
                    if j == 1:
                        xticks_coord.append([((i+0.75)+ 0.25*j)+i])#fetch wheel1 position
                except ValueError:
                    if j == 1:
                        xticks_coord.append([((i+0.75)+ 0.25*j)+i])#fetch wheel1 position
                    continue         
        #define legend
        l1, l2, l3 = [lines.Line2D((1,0), (0,1), marker = 'o', linestyle = 'None', color = colors[i], markeredgecolor = '#CFD0D1', ms = 4.0, figure = fig) for i in xrange(3)]
        fig.legend((l1, l2, l3), labels, fontsize = 10)
        #leg.get_frame().set_linewidth(0.0)     
        #set ticks and labels
        ax.set_xticks(xticks_coord)
        ax.set_xticklabels(batchkeys, rotation = 60, fontsize = 11)
        axis = ['x', 'y']
        [ax.tick_params(axis = item, labelsize = 9) for item in axis]
        #set axis and fig titles
        ax.set_ylabel('Time (hours)')
        ax.set_title('%s screen: KM median survival estimates' %tablename, fontsize = 13)
        plt.tight_layout()
        plt.show()
        return
    
    
    def overlayToD(self, speed = 33, output = 'screen'):
        from collections import OrderedDict
        import cv2
        #Fetch stockdata
        try:
            stockdata = self.fetchStockdata()
            if len(stockdata)==0:
                sys.exit()
        except AssertionError, e:
            print(e)
            sys.exit()
        #load plate lists
        [plateFlyTracks, todplate] = stockdata[0]
        #build todFrame dictionary
        well_labels, frameN, censor, tod, survPerc = zip(*todplate)
        todFrameDict = OrderedDict(zip(well_labels, frameN))
        #load well coordinates dictionary
        xywellMap = self.xywellMap()   
        #fetch image sequence
        filenamelist, imgpaths = self.loadPlateImgSeq()
        todFramenumber = [todFrameDict[key] for key in todFrameDict.keys() if key != 't0']
        #overlay circles over wells
        for i, path in enumerate(imgpaths):
                img = cv2.imread(path, 1)
                if (i+1) in todFramenumber:
                    wellId = [todFrameDict.keys()[j+1] for j, idx in enumerate(todFramenumber) if idx == (i+1)]
                    xy = [xywellMap[well] for well in wellId]
                    xy = [tuple([int(val) for val in tupl]) for tupl in xy]
                    [cv2.circle(img, tupl, 20, (255, 0, 255), thickness=3, lineType=8, shift=0) for tupl in xy]
                if output == 'file' or output == 'both':
                    imgseqdir = os.path.join(self.stockdir, 'OverlayToD')
                    if not os.path.exists(imgseqdir):
                        os.mkdir(imgseqdir)
                    imgpath = os.path.join(imgseqdir, filenamelist[i])
                    print('Saving overlayed image: %s\n' %imgpath)    
                    cv2.imwrite(imgpath, img)
                elif output == 'screen' or output == 'both':
                    cv2.imshow('image',img)
                    cv2.waitKey(speed)      
        cv2.waitKey(0)            
        cv2.destroyAllWindows()        
        return
        
    
    def overlayTrackingOnPlate(self):
        import cv2
        #Fetch stockdata
        try:
            stockdata = self.fetchStockdata()
            if len(stockdata)==0:
                sys.exit()
        except AssertionError, e:
            print(e)
            sys.exit()
        #load plate lists
        [plateFlytracks, todplate] = stockdata[0]
        trackdata = plateFlytracks.values()
        framenames = plateFlytracks.keys()
        originalImg, originalImg_paths = self.loadPlateImgSeq()
        #build index of well_labels in relation to self.wellIds
        well_labels, frameN, censor, tod, survPerc = zip(*todplate[1:])
        #sweep through frames
        for i, img in enumerate(trackdata):
            print('Overlaying frame_%03d' %(i+1))
            [frameContours, wellsMapOfContours_filtered, binTracker] = img
            original = cv2.imread(originalImg_paths[i], 1)
            #overlay bounding rectangles on contours
            for well in wellsMapOfContours_filtered:
                #test whether well contains a contour
                if len(well)>0:
                    (wellId, cnt, z, mean, centroid) = well
                    #idx = self.wellIds.index(wellId)
                    idx = well_labels.index(wellId)
                    #draw bounding rectangles only in wells with live flies
                    if frameN[idx] < i+1:
                        continue                  
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 1)    
                else:
                    continue   
            #save images    
            overlaytrackdir = os.path.join(self.stockdir, 'OverlayTracking')
            if not os.path.exists(overlaytrackdir):
                os.makedirs(overlaytrackdir)
            outpath = os.path.join(overlaytrackdir, '%s.jpg' %framenames[i])
            cv2.imwrite(outpath, original)
        return
        
    
    def showPlateTracking(self, output = 'screen'):
        import cv2
        #set variables
        counter = 0
        #inititate videocapture
        overlaytrackdir = os.path.join(self.stockdir, 'OverlayTracking')
        if not os.path.exists(overlaytrackdir):
            raise IOError('The %s path does not exists. \n' %overlaytrackdir)
        filepath = os.path.join(overlaytrackdir, 'frame_001.jpg')    
        cap = cv2.VideoCapture(filepath)
        if output == 'screen':
            winName = 'Plate tracker'
            cv2.namedWindow(winName)
        while(1):
            counter +=1
            ret, frame = cap.read()
            #test whether frame is empty 
            if np.count_nonzero(frame) == 0:
                break
            if output == 'screen':
                cv2.imshow(winName, frame)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            elif output == 'file':
                if counter == 1:
                    height , width , layers =  frame.shape
                    videopath = os.path.join(self.stockdir, '%s_tracked.avi' %self.fwId)
                    # Define the codec and create VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    writerObj = cv2.VideoWriter(videopath, fourcc, 20.0,(width,height))
                #write file
                writerObj.write(frame)
        cap.release()
        cv2.destroyAllWindows()
        return
        
        
    def showCensoredWells(self):
        import cv2
        #load data objects
        wellsmap = self.xywellMap()
        censorRegistry = self.loadWellCensorRegistry()
        #load image    
        filelist, imgpathlist = self.loadPlateImgSeq()
        image = cv2.imread(imgpathlist[0], 0)
        #fetch censored wells IDs
        try:
            censoredWells = censorRegistry[self.fwId]
        except KeyError:
            print('%s: no wells were censored' %self.fwId)
        #display censored wells on image
        coordinates = [wellsmap[wellId] for wellId in censoredWells]
        [cv2.circle(image, (int(x),int(y)), 10, (255,255,255), 2) for (x,y) in coordinates]
        cv2.imshow(self.fwId, image)
        cv2.waitKey(0)
        return




#custompath = 'U:\Flywheel\Data\OtherUsers\David'
#fwId = ['JS10_01082012_24', 'JS10_28022013_502']
#dv = DataVis(fwId)
#dv.plotDataset('Starvation', showstock = 'JS208', showassay =False)
#dv.writePlateToD()
#dv.overlayTrackingOnPlate()
#dv.showPlateTracking()
#dv.survPlotter()
#dv.stockMedSurv('ROS', 'JS10', plottype = 'swarmplot')
#custompath = 'U:\Flywheel\Data\OtherUsers\David\Run2'
#fwId = 'FW2105-1'
#dv = DataVis(fwId, custompath = custompath)
#dv.overlayTrackingOnPlate()
#dv.showPlateTracking(output = 'file')
#dv.survPlotter()
#plt = PlateTracker(fwId, custompath = custompath)
#plt.plateToDCrawler()
#plt.plateAnalyser(period = 30, output = 'remote')
#plt.plateFlytracker(output = 'local')
#dv.platePlotter(todlabel = 'time')
#dirpath = 'U:\Flywheel\Data\Starvation\Assays'
#Dashboard().batchPlateAnalyser(dirpath, timelabel = 'timestamp', output = 'remote')
#dv.writePlateToD()
#dv.showCensoredWells()




