
#python code for geochemical anomaly extraction
#Written By:
#Bowen Chen and Yongliang Chen

import numpy as np
import matplotlib.pyplot as plt
import math
import struct
import os
import copy
from time import time
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import mixture
from sklearn.semi_supervised import LabelPropagation


os.chdir(r'd:\\Chengde')


class FileInputOutput:

    def __init__(self,k,l,bins,deposits,xmin,xmax,ymin,ymax,inputfile,permiterror = 0.0001):
        self.filename = inputfile
        self.k = k
        self.l = l
        self.bins = bins
        self.maxdeposits = deposits#500
        self.permiterror = permiterror#0.0001
        self.type = "Youden"
        #self.type = "likelihoodratio"
        #self.type = "lift"
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def  construct_griddata_from_grid_Files(self):

        namefile = open(self.filename)
        i = 0
        ip = 0
        ii = 0
        iii = 0
        igc = 0

        for line in namefile.readlines():
            i += 1

            if line.find("mineral occurrence map:") == 0:
                ii = i+1
            if i == ii:
                #Open mineral deposit file and input the data
                string,s = line.split(".")
                string += ".mif"
                deposit_num,deposit_co = self.input_deposit_data(string)
                #print string
                dx,dy = self.generate_gridcells(self.xmin,self.xmax,self.ymin,self.ymax)
                deposit = np.zeros(self.k*self.l,float)
                deposit = self.determine_orebearing_of_gridcells(deposit_co,deposit_num,self.xmin,self.ymin,dx,dy)
            if line.find("geochemical maps:") == 0:
                iii = i+1
            if i == iii:
                igc = int(line)
                print ("The number of grid files:",igc)
                grid_attribute = np.zeros((self.k*self.l,igc),float)
                auc = np.zeros(igc,float)
                youden = np.zeros(igc,float)
                
            if i > iii and i <= iii + igc:
                ip += 1
                string,s = line.split(".")
                string += ".grd"
                #Open binery grid file and input grid data
                ncases,linNum,colNum,xmin1,xmax1,ymin1,ymax1,zmin,zmax,data = self.input_binary_grid_file(string)
                auc[ip-1] = self.test_deposit_correlation(ncases,deposit,data)
                youden[ip-1],roc = self.extract_geochemical_anomaly(ncases,zmin,zmax,data,deposit)
                s1,s2 = string.split(".")
                s1 += "roc.txt"
                #print s1
                #Output the roc data into a text file
                self.output_roc_data(s1,roc)

                #Data Logarithmic Transformation
                #zmin,zmax,data = self.log_data(ncases,zmin,zmax,data)
                
                #Data Normalization               
                grid_attribute[:,ip-1] = self.normalize_data(ncases,zmin,zmax,data)

                #Data Standardization
                #grid_attribute[:,ip-1] = self.standardize_data(ncases,data)
                
              

        namefile.close()
        
        #Preprocessing grid_attribute data
        #ndim = m+1
        #grid_attribute = self.preprocessing_grid_attribute(ncases,ndim,grid_attribute)
        
        #Output the grid_attribute data into a text file
        #self.output_grid_attribute_data("Attribute_data.txt",ncases,m,self.l,self.k,xmin,xmax,ymin,ymax,grid_attribute)
         
        return deposit,grid_attribute,auc,youden,zmax,ncases,igc,self.k,self.l


    # Read grid file in binary format
    def input_binary_grid_file(self,filename):

        binaryfile = open(filename,"rb")

        HeadSign = binaryfile.read(4)
        print (HeadSign.decode("utf-8"))
        
        if HeadSign.decode("utf-8") == "DSBB":
            colNum, = struct.unpack("h",binaryfile.read(2))
            linNum, = struct.unpack("h",binaryfile.read(2))
            xmin, = struct.unpack("d",binaryfile.read(8))    
            xmax, = struct.unpack("d",binaryfile.read(8))
            ymin, = struct.unpack("d",binaryfile.read(8))
            ymax, = struct.unpack("d",binaryfile.read(8))
            zmin, = struct.unpack("d",binaryfile.read(8))
            zmax, = struct.unpack("d",binaryfile.read(8))
        
            ncases = colNum*linNum
            data = np.zeros(ncases,float)
            
            for i in range(ncases):
                x, = struct.unpack("f",binaryfile.read(4))
                data[i] = x
        elif HeadSign.decode("utf-8") == "DSRB":
            HeadLength, = struct.unpack("l",binaryfile.read(4)) 
            Version, = struct.unpack("l",binaryfile.read(4))
            GridSign = binaryfile.read(4)
            print (GridSign.decode("utf-8"))
            GridLength, = struct.unpack("l",binaryfile.read(4))
            linNum, = struct.unpack("l",binaryfile.read(4))
            colNum, = struct.unpack("l",binaryfile.read(4))
            xmin, = struct.unpack("d",binaryfile.read(8))    
            ymin, = struct.unpack("d",binaryfile.read(8))
            xsize, = struct.unpack("d",binaryfile.read(8))
            ysize, = struct.unpack("d",binaryfile.read(8))
            zmin, = struct.unpack("d",binaryfile.read(8))
            zmax, = struct.unpack("d",binaryfile.read(8))
            Rotation, = struct.unpack("d",binaryfile.read(8))
            BlankValue, = struct.unpack("d",binaryfile.read(8))
            DataSign = binaryfile.read(4)
            print (DataSign.decode("utf-8"))
            DataLength, = struct.unpack("l",binaryfile.read(4))
            
            ncases = colNum*linNum
            data = np.zeros(ncases,float)
            
            for i in range(ncases):
                x, = struct.unpack("d",binaryfile.read(8))
                data[i] = x
            xmax = xmin + xsize*(colNum-1)
            ymax = ymin + ysize*(linNum-1)        
        binaryfile.close()
        
        print (filename)
        print ("Data information:")
        print ('Samples:',ncases)
        print
        print ("Grid information:")
        print ('Columns:',colNum,',','Rows:',linNum)
        print
        print ("Coordinates:")
        print ('xmin=',xmin,',','xmax=',xmax)
        print ('ymin=',ymin,',','ymax=',ymax)
        print ('zmin=',zmin,',','zmax=',zmax)
        print
        
        """
        for i in range(ncases):
            if (i+1)%10 == 0 or (i+1)%colNum == 0:
                print (data[i],"\n")
            else:
                print (data[i],)
        """   
        return ncases,linNum,colNum,xmin,xmax,ymin,ymax,zmin,zmax,data

    def test_deposit_correlation(self,ncases,deposit,data):
    
        na = 0
        nb = 0
        for i in range(ncases):
            if data[i] < 1.70141E+038:
                if deposit[i] == 1.0:
                    na += 1
                else:
                    nb += 1
        #print "na =",na,"nb=",nb
            
        area = 0.0
        for i in range(ncases):
            if data[i] < 1.70141E+038:
                if deposit[i] == 1.0:
                    for j in range(ncases):
                        if data[j] < 1.70141E+038:
                            if deposit[j] == 0.0:
                                if data[i] > data[j]:
                                    area += 1.0
                                elif data[i] == data[j]:
                                    area += 0.5
                                else:
                                    continue
        area = area/float(na*nb)        
        q1 = area/(2.0-area)
        q2 = 2.0*area*area/(1.0+area)
        se = math.sqrt((area*(1-area)+(na - 1.0)*(q1 - area*area)+(nb-1.0)*(q2 - area*area))/(na*nb))
        z = (area - 0.5)/se
        
        print ("Area under roc curve =",area) 
        print ("Standard error of area =",se)          
        print ("Test statistics =",z)
        
        return area

    def extract_geochemical_anomaly(self,ncases,zmin,zmax,data,deposit):

        roc = np.zeros((self.bins+2,2),float)
        roc[0,0] = 0.0
        roc[0,1] = 0.0
        roc[self.bins+1,0] = 1.0
        roc[self.bins+1,1] = 1.0
     
        delta = (zmax - zmin)/float(self.bins)
        maxcontrast = -99999.0
        optimalThrsh = -99999.0
        
        for j in range(1,self.bins):
            tp = 0.0
            fp = 0.0
            tn = 0.0
            fn = 0.0

            threshold = zmin + delta*(bins - j)
            for i in range(ncases):
                if data[i] < 1.70141E+038:
                    if deposit[i] == 1.0 and data[i] >= threshold:
                        tp += 1.0
                    if  deposit[i] == 0.0 and data[i] >= threshold:
                        fp += 1.0
                    if deposit[i] == 0.0 and data[i]< threshold:
                        tn += 1.0
                    if  deposit[i] == 1.0 and data[i] < threshold:
                        fn += 1.0
            
            fpf = fp/(fp+tn)
            tpf = tp/(tp+fn)
            roc[j,0] = fpf
            roc[j,1] = tpf
            
            if self.type == "Youden":
                contrast = tpf - fpf
                if contrast > maxcontrast:
                    maxcontrast = contrast
                    optimalThresh = threshold
            elif self.type == "likelihoodratio":
                if fpf != 0.0:
                    contrast = tpf / fpf
                    if contrast > maxcontrast:
                        maxcontrast = contrast
                        optimalThresh = threshold
            else:
                pyrate = (tp+fp)/(tp+fn+tn+fp)
                lift = tp*(tp+fn+tn+fp)/((tp+fn)*(tp+fp))
                if lift > maxcontrast:
                    maxcontrast = lift
                    optimalThresh = threshold
        
        if self.type == "Youden":
            print ("Maximum Youden index =",maxcontrast)
        elif self.type == "likelihoodratio":
            print ("Maximum likelihoodratio =",maxcontrast)
        else:
            print ("Maximum lift =",maxcontrast)
        print ("Optimal Threshold =",optimalThresh)
        print ("maximum value =", zmax)
        
        """
        binary = np.zeros(ncases,float)
            
        for i in range(ncases):
            if data[i] > optimalThresh and data[i] < 1.70141E+038:
                 binary[i] = 1.0
            else:
                 binary[i] = 0.0
        """
        return maxcontrast,roc

    def input_deposit_data(self,filename):

         datafile = open(filename)
         deposit_co = np.zeros((self.maxdeposits,2),float)

         #Input the data from the *.mid file(including the deposit data)
         i = 0
         for line in datafile.readlines():
              """
              if line.find("CoordSys") == 0:
                   string1,string2,string3 = line.split("(")
                   string4,string5 = string2.split(")")
                   xmin,ymin = string4.split(", ")
                   xmin = float(xmin)
                   ymin = float (ymin)
                   string6,string7 = string3.split(")")
                   xmax,ymax = string6.split(", ")
                   xmax = float(xmax)
                   ymax = float(ymax)
              """
              if line.find("Point") == 0:
                   string,xcor,ycor = line.split(" ")
                   deposit_co[i, 0] = float(xcor)
                   deposit_co[i, 1] = float(ycor)
                   i += 1

         deposit_num = i
         
         print ("xmin =",xmin,"ymin=",ymin)
         print ("xmax =",xmax,"ymax =",ymax)
         print ("deposit_num =",i)
         
         for i in range(deposit_num):
              print (deposit_co[i,0],deposit_co[i,1])
         
         datafile.close()
         
         return deposit_num,deposit_co

    def generate_gridcells(self,xmin,xmax,ymin,ymax):
          
        """
        double ccc = (xmax - xmin) / (ymax - ymin)
        
        #The number of columns
        l = int(self.k * ccc)
        if l != 200:
            print "Error!, The number of collumn is wrong!"
            exit(0)
        """
        dx = (xmax - xmin) / float(self.l)
        dy = (ymax - ymin) / float(self.k)

        """
        grid_co = np.zeros((self.k * self.l, 2),float)
        
        #Generate the uniform grid cells
        for i in range(self.k):
            for j in range(self.l):
                grid_co[i*self.l+j, 0] = xmin + j * dx + dx / 2.0
                grid_co[i*self.l+j, 1] = ymin + i * dy + dy / 2.0

        print "k =",self.k,"l=",self.l
        print "ncases=", self.k*self.l
        """
        return dx,dy


    def determine_orebearing_of_gridcells(self,deposit_co,deposit_num,xmin,ymin,dx,dy):

        griddata = np.zeros(self.k*self.l,float)
        #Determining grid cells whether containning a deposit
         
        for i in range (deposit_num):
            ii = int((deposit_co[i,0] - xmin)/dx)
            jj = int((deposit_co[i,1] - ymin)/dy)
            griddata[jj*self.l+ii] = 1.0

        return griddata
    
    def log_data(self,ncases,minVal,maxVal,data):

        #print ("Logarithmic Transforming Data...")
        minVal = math.log(minVal)
        minVal = math.log(maxVal)
       
        for i in range(0,ncases):
            if data[i] < 1.70141E+038:
                data[i] = math.log(data[i])
        return minVal,maxVal,data
       
    def normalize_data(self,ncases,minVal,maxVal,data):

        #print ("Normalizing Data...")
       
        for i in range(0,ncases):
            if data[i] < 1.70141E+038:
                data[i] = (data[i]-minVal)/(maxVal-minVal)
        return data

    def standardize_data(self,ncases,data):

        #print ("Standardizizing Data...")
        np.std(data, axis=0)
        np.mean(data, axis=0)

        sample = 0
        sum1 = 0.0
        sum2 = 0.0
        for i in range(0,ncases):
            if data[i] < 1.70141E+038:
                sum1 += data[i]
                sum2 += data[i]**2
                sample += 1

        mean = sum1/float(sample)
        dev  = math.sqrt(sum2/float(sample) - (sum1**2/(float(sample))**2))

        for i in range(0,ncases):
            if data[i] < 1.70141E+038:
                data[i] = (data[i] - mean)/dev
        
        return data
    
    def weighted_combination(self,ncases,m,auc,deposit,data):

        wdata = np.ones(ncases,float)* 1.70141E+38
        sume = auc.sum()
        for i in range(m):
            auc[i] = auc[i]/sume

        zmin = 999999.0
        zmax = -999999.0
        for i in range(ncases):
            #identify non-blank areas
            sume = 0
            for j in range(m):
                if data[i,j] < 1.70141E+38:
                    sume += 1
            if sume == m:
                
                ss = 0.0
                for jj in range(m):
                    ss += auc[jj]* data[i,jj]
                if ss > zmax:
                    zmax = ss
                if ss < zmin:
                    zmin =ss
                wdata[i] = ss
                """
                wdata[i] = data[i,:].sum()/float(m)
                """
        area = self.test_deposit_correlation(ncases,deposit,wdata)
        #youden,roc = self.extract_geochemical_anomaly(ncases,zmin,zmax,wdata,deposit)
        #print "zmin =",zmin,"zmax =",zmax
        #Output the roc data into a text file
        #self.output_roc_data("WeightedDataRoc.txt",roc)
        return wdata,area,zmax
                
    def comput_histgram(self,ncases,zmin,zmax,data):

        hist = np.zeros((self.bins+1,2),float)

        delta = (zmax - zmin)/float(self.bins)
        
        for j in range(self.bins + 1):
            hist[j,0] = zmin + delta * j
            
        sumj = 0.0    
        for j in range(self.bins + 1):
            for i in range(ncases):
                if data[i] < 1.70141E+038:
                    if  data[i] >= hist[j,0]- delta/2.0 and data[i] < hist[j,0] + delta/2.0:
                        hist[j, 1] += 1.0
            sumj += hist[j,1]
                        
        
        for j in range(self.bins + 1):
            hist[j,1] = 100.0*hist[j,1]/sumj
        
        
        return hist

    def comput_mean(self,ncases,data):

        sample = 0
        sumj = 0.0    
        for i in range(ncases):
            if data[i] < 1.70141E+038:
                sumj += data[i]
                sample += 1
        mean = sumj / sample

        sumj = 0.0
        for i in range(ncases):
            if data[i] < 1.70141E+038:
                sumj += (data[i] - mean)*(data[i] - mean)       
        variance = math.sqrt(sumj/sample)
        threshold = mean + 2.0*variance
        
        return mean, variance, threshold
    
    def comput_likeihood(self,ncases,data,deposit,threshold):
        
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0

        for i in range(ncases):
            if data[i] < 1.70141E+038:
                if deposit[i] == 1.0 and data[i] > threshold:
                    tp += 1.0
                if  deposit[i] == 0.0 and data[i] > threshold:
                    fp += 1.0
                if deposit[i] == 0.0 and data[i]<= threshold:
                    tn += 1.0
                if  deposit[i] == 1.0 and data[i] <= threshold:
                    fn += 1.0
        
        fpf = fp/(fp+tn)
        tpf = tp/(tp+fn)
        if fpf != 0:
            contrast = tpf / fpf
       
        return contrast       
    

    
    def AnomalyPercent(self,ncases,threshold,data,deposit):
        
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0
        for i in range(ncases):
            if data[i] < 1.70141E+038:
                if deposit[i] == 1.0 and data[i] > threshold:
                    tp += 1.0
                if  deposit[i] == 0.0 and data[i] > threshold:
                    fp += 1.0
                if deposit[i] == 0.0 and data[i]<= threshold:
                    tn += 1.0
                if  deposit[i] == 1.0 and data[i] <= threshold:
                    fn += 1.0
            
        fpf = fp/(fp+tn)
        tpf = tp/(tp+fn)
        
        print ("youden  =", tpf - fpf)
        print ("percenta =", tpf*100.0)
        print ("percentt =",(tp+fp)*100.0/(tp+fp+tn+fn))
        


    def input_grid_file(self,filename):

        textfile = open(filename)
        line_num = 0
        l = 0
        for line in textfile.readlines():
            if line_num == 1:
                colNum,linNum = line.split(" ")
                colNum = int(colNum)
                linNum = int(linNum)
                ncases = colNum * linNum
                data = np.zeros(ncases,float)
                print ("colNum =",colNum,"linNum =",linNum)
            elif line_num == 2:
                xmin,xmax = line.split(" ")
                print ("xmin =",xmin,"xmax =",xmax)
            elif line_num == 3:
                ymin,ymax = line.split(" ")
                print ("ymin =",ymin,"ymax =",ymax)
            elif line_num == 4:
                zmin,zmax = line.split(" ")
                print ("zmin =",zmin,"zmax =",zmax)
            elif line_num >= 5:
                ii = len(line.split(" "))
                #print "ii =",ii
                data[l:l+ii] = line.split(" ")
                l = l + ii
            line_num += 1
        textfile.close()
        """
        for i in range(linNum):
            for j in range (colNum):
                if (j+1)%10 == 0 or j+1 == colNum:
                    print data[i*colNum +j],
                    print
                else:
                    print data[i*colNum +j],                
        """
        return ncases,linNum,colNum,xmin,xmax,ymin,ymax,zmin,zmax,data



    def output_grid_file(self,filename,colNum,linNum,xmin,xmax,ymin,ymax,result):

        gridfile = open(filename,"w")

        zmin = result.min()
        
        ncases = linNum*colNum
        zmax = -999999.0
        for i in range(ncases):
            if result[i]> zmax and result[i] < 1.70141E+038:
                zmax = result[i]

        gridfile.write("DSAA\n")
        gridfile.write(str(colNum))
        gridfile.write (" ")
        gridfile.write(str(linNum))
        gridfile.write("\n")
        gridfile.write(str(xmin))
        gridfile.write(" ")
        gridfile.write(str(xmax))
        gridfile.write("\n")
        gridfile.write(str(ymin))
        gridfile.write(" ")
        gridfile.write(str(ymax))
        gridfile.write("\n")
        gridfile.write(str(zmin))
        gridfile.write(" ")
        gridfile.write(str(zmax))
        gridfile.write("\n")
        
        for i in range(linNum):
            for j in range(colNum):
                if (j+1)%10 == 0 or j == colNum-1:
                    gridfile.write(str(result[i*colNum+j]))
                    gridfile.write("\n")
                else:
                    gridfile.write(str(result[i*colNum+j]))
                    gridfile.write(" ")
        gridfile.close()


    def output_roc_data(self,filename,roc):

        textfile = open(filename,"w")
        textfile.write(str(roc[0,0]))
        textfile.write(",")
        textfile.write(str(roc[0,1]))
        textfile.write(",")
        for ir in range(1,self.bins):
            textfile.write(str(roc[ir,0]))
            textfile.write(",")
            textfile.write(str(roc[ir,1]))
            textfile.write("\n")
        textfile.write(str(roc[self.bins+1,0]))
        textfile.write(",")
        textfile.write(str(roc[self.bins+1,1]))
        textfile.write("\n")
        textfile.close()

class PrePostprocess:
    
    def __init__(self):

        print


    def preprocess_data(self,ncases,m,data,deposit):

        indicate = np.zeros(ncases,int)

        sample = 0        
        for i in range(ncases):
            sume = 0

            for j in range(m):
                if data[i,j] < 1.70141E+38:
                    sume += 1
            if sume == m:
                indicate[i] = 1
                sample += 1
                
        data1 = np.zeros((sample,m),float)
        dep = np.zeros(sample,float)

        ii = 0
        for i in range(ncases):
            if indicate[i] == 1:
                data1[ii,:] = data[i,:]
                dep[ii] = deposit[i]
                ii += 1

        return indicate,data1,dep,sample
    

    def postprocess_data(self,ncases,d,result):


        distance = np.ones(ncases,float)*1.70141E+38

        #restore the original data
        ii = 0
        for i in range (ncases):
            if d[i] == 1:
                distance[i] = result[ii]
                ii += 1

        return distance
    

        
class ROC_Analysis:

    def __init__(self):

        print ("ROC Analysis....")
        
    def compute_roc(self,bins,ncases,d,deposit,prob,textfilenamer,textfilenameg,textfilenamel):

        print ("compute roc curve...")
        roc = np.zeros((bins+1,2),float)
        gain = np.zeros((bins+1,2),float)
        lift = np.zeros((bins+1,2),float)
        
        LR = np.zeros(bins+1,float)
        youden = np.zeros(bins+1,float)

       
        p = int(deposit.sum())
        sample = 0
        for i in range(ncases):
            if d[i] == 1:
                sample += 1
                
        n = sample - p
        
        for j in range (2):
            roc[bins,j] = 1.0
            gain[bins,j] = 1.0
        lift[bins,0] = gain[bins,0]
        lift[bins,1] = 1.0/(1.0 + math.exp(-1.0))    
        #lift[0,1] = gain[0,0]
        
        minprob = prob.min()
        maxprob = -9999999.0
        for i in range (ncases):
            if d[i] == 1:
                if prob[i]> maxprob:
                    maxprob = prob[i]

        
        delta = (maxprob - minprob)/float(bins)
        maxyouden = -99999.0
        maxLR = -99999.0
        
        for j in range(1,bins):
            tp = 0.0
            fp = 0.0
            tn = 0.0
            fn = 0.0

            threshold = minprob + delta*(bins - j)
            for i in range(ncases):
                if d[i] == 1:
                    if prob[i] >= threshold and deposit[i] == 1.0:
                        tp += 1.0
                    if  prob[i] >= threshold and deposit[i] == 0.0:
                        fp += 1.0
                    if prob[i] < threshold and deposit[i] == 0.0:
                        tn += 1.0
                    if  prob[i] < threshold and deposit[i] == 1.0:
                        fn += 1.0
                    
            roc[j,0]= fp/(fp+tn)
            roc[j,1]= tp/(tp+fn)
            gain[j,0]= (tp+fp)/(p+n)#*100.0 #yrate
            gain[j,1]= roc[j,1]#*100.0
            lift[j,0]= gain[j,0]
            if gain[j,0] != 0.0:
                lift[j,1]= 1.0/(1.0 + math.exp(-gain[j,1]/gain[j,0]))
            
            youden[j] = roc[j,1] - roc[j,0]

            if youden[j] > maxyouden:
                maxyouden = youden[j]
                besthreshold = threshold
            
            if  roc[j,0] != 0:
                LR[j] =  1.0/(1.0 + math.exp(-roc[j,1]/roc[j,0]))
                if LR[j] > maxLR:
                    maxLR = LR[j]
                    besthresholdLR = threshold        
        asum = 0.0
        for i in range(ncases):
            if d[i] == 1:
                if prob[i] >= besthreshold:
                    asum = asum +1.0
        asum = asum/float(ncases)        
                

        print (textfilenamer,":")
        print ("maximumYoudenIndex =",maxyouden,"besthreshold =",besthreshold, "Percent of target area =", asum)
        #print "maximumLR =",maxLR,"besthresholdLR =",besthresholdLR
        
        textfile = open(textfilenamer,"w")
        for j in range(bins+1):
           textfile.write(str(minprob + delta*(bins - j)))
           textfile.write(",")
           for i in range(2):
              textfile.write(str(roc[j,i]))
              textfile.write(",")
           textfile.write(str(gain[j,0]))
           textfile.write(",")
           textfile.write(str(youden[j]))
           textfile.write(",")
           textfile.write(str(LR[j]))
           textfile.write("\n")
        textfile.close()
        
        textfile = open(textfilenameg,"w")
        for j in range(bins+1):
           #textfile.write(str(minprob + delta*(bins - j)))
           #textfile.write(",")
           textfile.write(str(gain[j,0]))
           textfile.write(",")
           textfile.write(str(gain[j,1]))          
           textfile.write("\n")
        textfile.close()

        textfile = open(textfilenamel,"w")
        for j in range(1,bins+1):
           #textfile.write(str(minprob + delta*(bins - j)))
           #textfile.write(",")
           textfile.write(str(lift[j,0]))
           textfile.write(",")
           textfile.write(str(lift[j,1])) 
           textfile.write("\n")
        textfile.close()
        
    def compute_roc_area(self,ncases,d,deposit,prob):
        print ("compute the area under roc curve...")

        sample = 0
        for i in range(ncases):
            if d[i] == 1.0:
                sample += 1                
        
        pd = deposit.sum()
        print ("The number of known deposits:",pd)
        
        na = 0
        area = 0.0
        for i in range(ncases):
            if d[i] == 1:
                if deposit[i] == 1.0:
                    na += 1
                    for j in range(ncases):
                        if d[j] == 1:
                            if deposit[j] == 0.0:
                                if prob[i] > prob[j]:
                                    area += 1.0
                                elif prob[i] == prob[j]:
                                    area += 0.5
                                else:
                                    continue
                            else:
                                continue
                else:
                    continue
        area = area/float(na*(sample-na))        
        q1 = area/(2.0-area)
        q2 = 2.0*area*area/(1.0+area)
        se = math.sqrt((area*(1-area)+(na - 1.0)*(q1 - area*area)+(sample-na-1.0)*(q2 - area*area))/(na*(sample-na)))
        z = (area - 0.5)/se
        aul = pd/(2*sample)+(1-pd/sample)*area

        return area,se,z,aul

# main procedure
if __name__ == '__main__':

    #Controlling parameters
    bins = 500
    deposits = 500#the maximum number of the input mineral deposits
    tolerance = 0
    
    #Map area (Longitude and Latitude)


    xmin = 117.7497205
    xmax = 118.2508928
    ymin = 40.83588747
    ymax = 41.16692999


    k = 180 #k is the number of rows
    l = 272 #int((xmax - xmin)/(ymax - ymin) * k + 0.5) #l is the number of collumns    
    
    print ("k =",k)
    print ("l =",l)
    
    #construct class objects
    inputfile = "DepositGridFileNameFe.txt"
    DataInAndOut = FileInputOutput(k,l,bins,deposits,xmin,xmax,ymin,ymax,inputfile)
    rocobj = ROC_Analysis()
    
    
    #DataInput
    deposit,data,auc,youden,zmax,ncases,ndim,linNum,colNum = DataInAndOut.construct_griddata_from_grid_Files()


    #Preprocessing 
    process = PrePostprocess()
    d,da,dep,sample = process.preprocess_data(ncases,ndim,data,deposit)
    
    
    #########################################################################################
    print ("LabelPropagation modeling...knn50")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    
    st99 = LabelPropagation(kernel='knn',gamma=0.01,n_neighbors=50,max_iter=3000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelknn50.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelknn50roc.txt","Labelknn50gain.txt","Labelknn50lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)


    
    #########################################################################################
    print ("LabelPropagation modeling...knn40")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    
    st99 = LabelPropagation(kernel='knn',gamma=0.01,n_neighbors=40,max_iter=3000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelknn40.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelknn40roc.txt","Labelknn40gain.txt","Labelknn40lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)
      
    #########################################################################################
    print ("LabelPropagation modeling...knn30")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    
    st99 = LabelPropagation(kernel='knn',gamma=0.01,n_neighbors=30,max_iter=3000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelknn30.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelknn30roc.txt","Labelknn30gain.txt","Labelknn30lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)


    #########################################################################################
    
    print ("LabelPropagation modeling...knn20")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    
    st99 = LabelPropagation(kernel='knn',gamma=0.01,n_neighbors=20,max_iter=3000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelknn20.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelknn20roc.txt","Labelknn20gain.txt","Labelknn20lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)
    

    #########################################################################################
    print ("LabelPropagation modeling...knn10")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    
    st99 = LabelPropagation(kernel='knn',gamma=0.01,n_neighbors=10,max_iter=10000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelknn10.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelknn10roc.txt","Labelknn10gain.txt","Labelknn10lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)

    
    #########################################################################################
    
    print ("LabelPropagation modeling...rbf001")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    
    st99 = LabelPropagation(kernel='rbf',gamma=0.01,max_iter=1000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelrbf001.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelrbf001roc.txt","Labelrbf001gain.txt","Labelrbf001lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)
    
    #########################################################################################
    print ("LabelPropagation modeling...rbf01")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    

    st99 = LabelPropagation(kernel='rbf',gamma=0.1,max_iter=1000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelrbf01.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelrbf01roc.txt","Labelrbf01gain.txt","Labelrbf01lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)

    #########################################################################################
    print ("LabelPropagation modeling...rbf05")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    

    st99 = LabelPropagation(kernel='rbf',gamma=0.5,max_iter=1000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelrbf05.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelrbf05roc.txt","Labelrbf05gain.txt","Labelrbf05lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)

    #########################################################################################
    print ("LabelPropagation modeling...rbf1")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    

    st99 = LabelPropagation(kernel='rbf',gamma=1.0,max_iter=1000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelrbf1.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelrbf1roc.txt","Labelrbf1gain.txt","Labelrbf1lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)


    #########################################################################################
    print ("LabelPropagation modeling...rbf5")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    

    st99 = LabelPropagation(kernel='rbf',gamma=5.0,max_iter=1000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelrbf5.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelrbf5roc.txt","Labelrbf5gain.txt","Labelrbf5lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)

    #########################################################################################
    print ("LabelPropagation modeling...rbf10")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    

    st99 = LabelPropagation(kernel='rbf',gamma=10,max_iter=1000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelrbf10.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelrbf10roc.txt","Labelrbf10gain.txt","Labelrbf10lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)

    #########################################################################################
    print ("LabelPropagation modeling...rbf20")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    

    st99 = LabelPropagation(kernel='rbf',gamma=20,max_iter=1000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelrbf20.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelrbf20roc.txt","Labelrbf20gain.txt","Labelrbf20lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)

    #########################################################################################
    print ("LabelPropagation modeling...rbf30")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    

    st99 = LabelPropagation(kernel='rbf',gamma=30,max_iter=1000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelrbf30.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelrbf30roc.txt","Labelrbf30gain.txt","Labelrbf30lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)

    #########################################################################################
    print ("LabelPropagation modeling...rbf40")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    

    st99 = LabelPropagation(kernel='rbf',gamma=40,max_iter=1000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelrbf40.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelrbf40roc.txt","Labelrbf40gain.txt","Labelrbf40lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)

    
    #########################################################################################
    print ("LabelPropagation modeling...rbf50")
    start_time = time()


    X = da
    y = dep.ravel()
    rng = np.random.RandomState(0)
    y_rand = rng.rand(sample)
    y_99 = np.copy(y)

    # set random samples to be unlabeled
    y_99[y_rand < 0.995] = -1.0

    for i in range(sample):
        if y[i] == 1.0:
            y_99[i] = 1.0
    

    st99 = LabelPropagation(kernel='rbf',gamma=50,max_iter=1000).fit(X, y_99)
    dist_st = st99.predict_proba(X)

    scores_st = process.postprocess_data(ncases,d,dist_st[:,1])    

            
    DataInAndOut.output_grid_file("Labelrbf50.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores_st)
    rocobj.compute_roc(1000,ncases,d,deposit,scores_st,"Labelrbf50roc.txt","Labelrbf50gain.txt","Labelrbf50lift.txt")
    areaLabelp,seLabelp,zLabelp,aulLabelp = rocobj.compute_roc_area(ncases,d,deposit,scores_st)
    print ("areaLabelp =", areaLabelp,"seLabelp =", seLabelp,"zLabelp =", zLabelp,"aulLabelp =",aulLabelp)

    end_time = time()

    print ("time =", end_time - start_time)

    
    #################################################################
    print ("GMM modeling......1")
    start_time = time()
    
    #d,da,sample = discomp.preprocess_data(ncases,ndim,data)

    
    clf = mixture.GaussianMixture(n_components=1, covariance_type='full',tol=1e-6,max_iter=2000, init_params='kmeans')
    print ("Component number = ",1)
    clf.fit(da)
    aic = clf.aic(da)
    bic = clf.bic(da)
    print ("aic =",aic,"  bic =",bic)
    
    score = clf.score_samples(da)
    scoremax = score.max()
    scoremin = score.min()
    print ("scoremax =",scoremax,"scoremin =",scoremin)

    gmm = np.ones(ncases,float)*1.70141E+38
    #restore the original data
    ii = 0
    for i in range (ncases):
        if d[i] == 1.0:
            gmm[i] = scoremax - score[ii]
            ii += 1

    DataInAndOut.output_grid_file("GMM_Anomaly1.grd",colNum,linNum,xmin,xmax,ymin,ymax,gmm)
    rocobj.compute_roc(1000,ncases,d,deposit,gmm,"GMM_AnomnalyRoc1.txt","GMM_Anomalygain1.txt","GMM_Anomalylift1.txt")
    areaGMM1,seGMM1,zGMM1,aulGMM1 = rocobj.compute_roc_area(ncases,d,deposit,gmm)
    print ("areaGMM1 =", areaGMM1,"seGMM1 =", seGMM1,"zGMM1 =", zGMM1,"aulGMM1 =",aulGMM1)
    
    end_time = time()
    print ("time =", end_time - start_time)
    
    #################################################################
    print ("GMM modeling......2")
    start_time = time()
    
    #d,da,sample = discomp.preprocess_data(ncases,ndim,data)

    
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full',tol=1e-6,max_iter=2000, init_params='kmeans')
    print ("Component number = ",2)
    clf.fit(da)
    aic = clf.aic(da)
    bic = clf.bic(da)
    print ("aic =",aic,"  bic =",bic)
    
    score = clf.score_samples(da)
    scoremax = score.max()
    scoremin = score.min()
    print ("scoremax =",scoremax,"scoremin =",scoremin)

    gmm = np.ones(ncases,float)*1.70141E+38
    #restore the original data
    ii = 0
    for i in range (ncases):
        if d[i] == 1.0:
            gmm[i] = scoremax - score[ii]
            ii += 1

    DataInAndOut.output_grid_file("GMM_Anomaly2.grd",colNum,linNum,xmin,xmax,ymin,ymax,gmm)
    rocobj.compute_roc(1000,ncases,d,deposit,gmm,"GMM_AnomnalyRoc2.txt","GMM_Anomalygain2.txt","GMM_Anomalylift2.txt")
    areaGMM2,seGMM2,zGMM2,aulGMM2 = rocobj.compute_roc_area(ncases,d,deposit,gmm)
    print ("areaGMM2 =", areaGMM2,"seGMM2 =", seGMM2,"zGMM2 =", zGMM2,"aulGMM2 =",aulGMM2)
    
    end_time = time()
    print ("time =", end_time - start_time)
   
    
    #################################################################
    print ("GMM modeling......3")
    start_time = time()
    
    #d,da,sample = discomp.preprocess_data(ncases,ndim,data)

    
    clf = mixture.GaussianMixture(n_components=3, covariance_type='full',tol=1e-6,max_iter=2000, init_params='kmeans')
    print ("Component number = ",3)
    clf.fit(da)
    aic = clf.aic(da)
    bic = clf.bic(da)
    print ("aic =",aic,"  bic =",bic)
    
    score = clf.score_samples(da)
    scoremax = score.max()
    scoremin = score.min()
    print ("scoremax =",scoremax,"scoremin =",scoremin)

    gmm = np.ones(ncases,float)*1.70141E+38
    #restore the original data
    ii = 0
    for i in range (ncases):
        if d[i] == 1.0:
            gmm[i] = scoremax - score[ii]
            ii += 1

    DataInAndOut.output_grid_file("GMM_Anomaly3.grd",colNum,linNum,xmin,xmax,ymin,ymax,gmm)
    rocobj.compute_roc(1000,ncases,d,deposit,gmm,"GMM_AnomnalyRoc3.txt","GMM_Anomalygain3.txt","GMM_Anomalylift3.txt")
    areaGMM3,seGMM3,zGMM3,aulGMM3 = rocobj.compute_roc_area(ncases,d,deposit,gmm)
    print ("areaGMM3 =", areaGMM3,"seGMM3 =", seGMM3,"zGMM3 =", zGMM3,"aulGMM3 =",aulGMM3)
    
    end_time = time()
    print ("time =", end_time - start_time)
    
   
    #################################################################
    print ("GMM modeling......4")
    start_time = time()
    
    #d,da,sample = discomp.preprocess_data(ncases,ndim,data)

    
    clf = mixture.GaussianMixture(n_components=4, covariance_type='full',tol=1e-6,max_iter=2000, init_params='kmeans')
    print ("Component number = ",4)
    clf.fit(da)
    aic = clf.aic(da)
    bic = clf.bic(da)
    print ("aic =",aic,"  bic =",bic)
    
    score = clf.score_samples(da)
    scoremax = score.max()
    scoremin = score.min()
    print ("scoremax =",scoremax,"scoremin =",scoremin)

    gmm = np.ones(ncases,float)*1.70141E+38
    #restore the original data
    ii = 0
    for i in range (ncases):
        if d[i] == 1.0:
            gmm[i] = scoremax - score[ii]
            ii += 1

    DataInAndOut.output_grid_file("GMM_Anomaly4.grd",colNum,linNum,xmin,xmax,ymin,ymax,gmm)
    rocobj.compute_roc(1000,ncases,d,deposit,gmm,"GMM_AnomnalyRoc4.txt","GMM_Anomalygain4.txt","GMM_Anomalylift4.txt")
    areaGMM4,seGMM4,zGMM4,aulGMM4 = rocobj.compute_roc_area(ncases,d,deposit,gmm)
    print ("areaGMM4 =", areaGMM4,"seGMM4 =", seGMM4,"zGMM4 =", zGMM4,"aulGMM4 =",aulGMM4)
    
    end_time = time()
    print ("time =", end_time - start_time)


    #################################################################
    print ("GMM modeling......5")
    start_time = time()
    
    #d,da,sample = discomp.preprocess_data(ncases,ndim,data)

    
    clf = mixture.GaussianMixture(n_components=5, covariance_type='full',tol=1e-6,max_iter=2000, init_params='kmeans')
    print ("Component number = ",5)
    clf.fit(da)
    aic = clf.aic(da)
    bic = clf.bic(da)
    print ("aic =",aic,"  bic =",bic)
    
    score = clf.score_samples(da)
    scoremax = score.max()
    scoremin = score.min()
    print ("scoremax =",scoremax,"scoremin =",scoremin)

    gmm = np.ones(ncases,float)*1.70141E+38
    #restore the original data
    ii = 0
    for i in range (ncases):
        if d[i] == 1.0:
            gmm[i] = scoremax - score[ii]
            ii += 1

    DataInAndOut.output_grid_file("GMM_Anomaly5.grd",colNum,linNum,xmin,xmax,ymin,ymax,gmm)
    rocobj.compute_roc(1000,ncases,d,deposit,gmm,"GMM_AnomnalyRoc5.txt","GMM_Anomalygain5.txt","GMM_Anomalylift5.txt")
    areaGMM5,seGMM5,zGMM5,aulGMM5 = rocobj.compute_roc_area(ncases,d,deposit,gmm)
    print ("areaGMM5 =", areaGMM5,"seGMM5 =", seGMM5,"zGMM5 =", zGMM5,"aulGMM5 =",aulGMM5)
    
    end_time = time()
    print ("time =", end_time - start_time)

    #################################################################
    print ("GMM modeling......6")
    start_time = time()
    
    #d,da,sample = discomp.preprocess_data(ncases,ndim,data)

    
    clf = mixture.GaussianMixture(n_components=6, covariance_type='full',tol=1e-6,max_iter=2000, init_params='kmeans')
    print ("Component number = ",6)
    clf.fit(da)
    aic = clf.aic(da)
    bic = clf.bic(da)
    print ("aic =",aic,"  bic =",bic)
    
    score = clf.score_samples(da)
    scoremax = score.max()
    scoremin = score.min()
    print ("scoremax =",scoremax,"scoremin =",scoremin)

    gmm = np.ones(ncases,float)*1.70141E+38
    #restore the original data
    ii = 0
    for i in range (ncases):
        if d[i] == 1.0:
            gmm[i] = scoremax - score[ii]
            ii += 1

    DataInAndOut.output_grid_file("GMM_Anomaly6.grd",colNum,linNum,xmin,xmax,ymin,ymax,gmm)
    rocobj.compute_roc(1000,ncases,d,deposit,gmm,"GMM_AnomnalyRoc6.txt","GMM_Anomalygain6.txt","GMM_Anomalylift6.txt")
    areaGMM6,seGMM6,zGMM6,aulGMM6 = rocobj.compute_roc_area(ncases,d,deposit,gmm)
    print ("areaGMM6 =", areaGMM6,"seGMM6 =", seGMM6,"zGMM6 =", zGMM6,"aulGMM6 =",aulGMM6)
    
    end_time = time()
    print ("time =", end_time - start_time)

    #################################################################
    print ("GMM modeling......7")
    start_time = time()
    
    #d,da,sample = discomp.preprocess_data(ncases,ndim,data)

    
    clf = mixture.GaussianMixture(n_components=7, covariance_type='full',tol=1e-6,max_iter=2000, init_params='kmeans')
    print ("Component number = ",7)
    clf.fit(da)
    aic = clf.aic(da)
    bic = clf.bic(da)
    print ("aic =",aic,"  bic =",bic)
    
    score = clf.score_samples(da)
    scoremax = score.max()
    scoremin = score.min()
    print ("scoremax =",scoremax,"scoremin =",scoremin)

    gmm = np.ones(ncases,float)*1.70141E+38
    #restore the original data
    ii = 0
    for i in range (ncases):
        if d[i] == 1.0:
            gmm[i] = scoremax - score[ii]
            ii += 1

    DataInAndOut.output_grid_file("GMM_Anomaly7.grd",colNum,linNum,xmin,xmax,ymin,ymax,gmm)
    rocobj.compute_roc(1000,ncases,d,deposit,gmm,"GMM_AnomnalyRoc7.txt","GMM_Anomalygain7.txt","GMM_Anomalylift7.txt")
    areaGMM7,seGMM7,zGMM7,aulGMM7 = rocobj.compute_roc_area(ncases,d,deposit,gmm)
    print ("areaGMM7 =", areaGMM7,"seGMM7 =", seGMM7,"zGMM7 =", zGMM7,"aulGMM7 =",aulGMM7)
    
    end_time = time()
    print ("time =", end_time - start_time)

    #################################################################
    print ("GMM modeling......8")
    start_time = time()
    
    #d,da,sample = discomp.preprocess_data(ncases,ndim,data)

    
    clf = mixture.GaussianMixture(n_components=8, covariance_type='full',tol=1e-6,max_iter=2000, init_params='kmeans')
    print ("Component number = ",8)
    clf.fit(da)
    aic = clf.aic(da)
    bic = clf.bic(da)
    print ("aic =",aic,"  bic =",bic)
    
    score = clf.score_samples(da)
    scoremax = score.max()
    scoremin = score.min()
    print ("scoremax =",scoremax,"scoremin =",scoremin)

    gmm = np.ones(ncases,float)*1.70141E+38
    #restore the original data
    ii = 0
    for i in range (ncases):
        if d[i] == 1.0:
            gmm[i] = scoremax - score[ii]
            ii += 1

    DataInAndOut.output_grid_file("GMM_Anomaly8.grd",colNum,linNum,xmin,xmax,ymin,ymax,gmm)
    rocobj.compute_roc(1000,ncases,d,deposit,gmm,"GMM_AnomnalyRoc8.txt","GMM_Anomalygain8.txt","GMM_Anomalylift8.txt")
    areaGMM8,seGMM8,zGMM8,aulGMM8 = rocobj.compute_roc_area(ncases,d,deposit,gmm)
    print ("areaGMM8 =", areaGMM8,"seGMM8 =", seGMM8,"zGMM8 =", zGMM8,"aulGMM8 =",aulGMM8)
    
    end_time = time()
    print ("time =", end_time - start_time)


    #################################################################
    print ("GMM modeling......9")
    start_time = time()
    
    #d,da,sample = discomp.preprocess_data(ncases,ndim,data)

    
    clf = mixture.GaussianMixture(n_components=9, covariance_type='full',tol=1e-6,max_iter=2000, init_params='kmeans')
    print ("Component number = ",9)
    clf.fit(da)
    aic = clf.aic(da)
    bic = clf.bic(da)
    print ("aic =",aic,"  bic =",bic)
    
    score = clf.score_samples(da)
    scoremax = score.max()
    scoremin = score.min()
    print ("scoremax =",scoremax,"scoremin =",scoremin)

    gmm = np.ones(ncases,float)*1.70141E+38
    #restore the original data
    ii = 0
    for i in range (ncases):
        if d[i] == 1.0:
            gmm[i] = scoremax - score[ii]
            ii += 1

    DataInAndOut.output_grid_file("GMM_Anomaly9.grd",colNum,linNum,xmin,xmax,ymin,ymax,gmm)
    rocobj.compute_roc(1000,ncases,d,deposit,gmm,"GMM_AnomnalyRoc9.txt","GMM_Anomalygain9.txt","GMM_Anomalylift9.txt")
    areaGMM9,seGMM9,zGMM9,aulGMM9 = rocobj.compute_roc_area(ncases,d,deposit,gmm)
    print ("areaGMM9 =", areaGMM9,"seGMM9 =", seGMM9,"zGMM9 =", zGMM9,"aulGMM9 =",aulGMM9)
    
    end_time = time()
    print ("time =", end_time - start_time)

    #################################################################
    print ("GMM modeling......10")
    start_time = time()
    
    #d,da,sample = discomp.preprocess_data(ncases,ndim,data)

    
    clf = mixture.GaussianMixture(n_components=10, covariance_type='full',tol=1e-6,max_iter=2000, init_params='kmeans')
    print ("Component number = ",10)
    clf.fit(da)
    aic = clf.aic(da)
    bic = clf.bic(da)
    print ("aic =",aic,"  bic =",bic)
    
    score = clf.score_samples(da)
    scoremax = score.max()
    scoremin = score.min()
    print ("scoremax =",scoremax,"scoremin =",scoremin)

    gmm = np.ones(ncases,float)*1.70141E+38
    #restore the original data
    ii = 0
    for i in range (ncases):
        if d[i] == 1.0:
            gmm[i] = scoremax - score[ii]
            ii += 1

    DataInAndOut.output_grid_file("GMM_Anomaly10.grd",colNum,linNum,xmin,xmax,ymin,ymax,gmm)
    rocobj.compute_roc(1000,ncases,d,deposit,gmm,"GMM_AnomnalyRoc10.txt","GMM_Anomalygain10.txt","GMM_Anomalylift10.txt")
    areaGMM10,seGMM10,zGMM10,aulGMM10 = rocobj.compute_roc_area(ncases,d,deposit,gmm)
    print ("areaGMM10 =", areaGMM10,"seGMM10 =", seGMM10,"zGMM10 =", zGMM10,"aulGMM10 =",aulGMM10)
    
    end_time = time()
    print ("time =", end_time - start_time)

    
    ############################################################################################    
    
    # Create and fit an logistic regression
    print ("Logistic regression modeling...")
    start_time = time()
    
    clf = LogisticRegression(random_state=0,solver='newton-cg',max_iter=2000).fit(da,dep)
    prob = clf.predict_proba(da)

    scores = process.postprocess_data(ncases,d,prob[:,1])

    DataInAndOut.output_grid_file("LGR.grd",colNum,linNum,xmin,xmax,ymin,ymax,scores)
    rocobj.compute_roc(bins,ncases,d,deposit,scores,"LGRRoc.txt","LGRgain.txt","LGRlift.txt")
    areaLGR,seLGR,zLGR,aulLGR = rocobj.compute_roc_area(ncases,d,deposit,scores)
    print ("areaLGR =", areaLGR,"seLGR =", seLGR,"zLGR =", zLGR,"aulLGR =",aulLGR)

    end_time = time()

    print ("time =", end_time - start_time)   
    
    
    ################################################################################################
    
    print ("OK!")






































