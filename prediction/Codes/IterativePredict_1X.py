import cv2
import numpy as np
import os
import sys
import argparse
import multiprocessing
import lxml.etree as ET
import warnings
import time
from PIL import Image
from glob import glob
from subprocess import call
from joblib import Parallel, delayed
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
from shutil import rmtree
import matplotlib.pyplot as plt
sys.path.append(os.getcwd()+'\Codes')

from IterativeTraining import get_num_classes
from get_choppable_regions import get_choppable_regions
from get_network_performance import get_perf

"""
Pipeline code to segment regions from WSI

"""

# define xml class colormap
xml_color = [65280, 65535, 255, 16711680, 33023]

def validate(args):
    # define folder structure dict
    dirs = {'outDir': args.base_dir + '\\' + args.project + args.outDir}
    dirs['txt_save_dir'] = '\txt_files\\'
    dirs['img_save_dir'] = '\img_files\\'
    dirs['mask_dir'] = '\wsi_mask\\'
    dirs['chopped_dir'] = '\originals\\'
    dirs['save_outputs'] = args.save_outputs
    dirs['modeldir'] = '\MODELS\\'
    dirs['training_data_dir'] = '\TRAINING_data\\'
    dirs['validation_data_dir'] = '\HOLDOUT_data\\'

    # find current iteration
    if args.iteration == 'none':
        iteration = get_iteration(args=args)
    else:
        iteration = int(args.iteration)

    # get all WSIs
    WSIs = []
    for ext in [args.wsi_ext]:
        WSIs.append(glob(args.base_dir + '' + args.project + dirs['validation_data_dir'] + '\*' + str(ext)))

    if iteration == 'none':
        print('ERROR: no trained models found \n\tplease use [--option train]')

    else:
        for iter in range(1,iteration+1):
            dirs['xml_save_dir'] = args.base_dir + '\\' + args.project + dirs['validation_data_dir'] + str(iter) + '_Predicted_XMLs\\'


            # check main directory exists
            make_folder(dirs['outDir'])

            if not os.path.exists(dirs['xml_save_dir']):
                make_folder(dirs['xml_save_dir'])

            print('working on iteration: ' + str(iter))

            with open(args.base_dir + '\\' + args.project + dirs['validation_data_dir'] + 'validation_stats.txt', 'a') as f:
                f.write('\niteration: \t'+str(iter)+'\n')
                f.write('\twsi\t\t\tsensitivity\t\t\tspecificity\t\t\tprecision\t\t\taccuracy\t\t\tprediction time\n')

            for wsi in WSIs:
                # predict xmls
                startTime = time.time()

                filename=dirs['xml_save_dir']+'\\'+ (wsi.split('\\')[-1]).split('.')[0] +'.xml'
                if not os.path.isfile(filename):
                    predict_xml(args=args, dirs=dirs, wsi=wsi, iteration=iter)

                predictTime = time.time() - startTime
                # test performance
                gt_xml = os.path.splitext(wsi)[0] + '.xml'
                predicted_xml = gt_xml.split('\\')
                predicted_xml = dirs['xml_save_dir'] + predicted_xml[-1]
                sensitivity,specificity,precision,accuracy = get_perf(wsi=wsi, xml1=gt_xml, xml2 = predicted_xml, args=args)

                with open(args.base_dir + '\\' + args.project + dirs['validation_data_dir'] + 'validation_stats.txt', 'a') as f:
                    f.write('\t'+wsi.split('\\')[-1]+'\t\t'+str(sensitivity)+'\t\t'+str(specificity)+'\t\t'+str(precision)+'\t\t'+str(accuracy)+'\t\t'+str(predictTime)+'\n')

        print('\n\n\033[92;5mDone validating: \n\t\033[0m\n')

def predict(args):
    # define folder structure dict
    dirs = {'outDir': args.base_dir + '\\' + args.project + args.outDir}
    dirs['txt_save_dir'] = '\\txt_files\\'
    dirs['img_save_dir'] = '\img_files\\'
    dirs['mask_dir'] = '\wsi_mask\\'
    dirs['chopped_dir'] = '\originals\\'
    dirs['save_outputs'] = args.save_outputs
    dirs['modeldir'] = '\MODELS\\'
    dirs['training_data_dir'] = '\TRAINING_data\\'

    # find current iteration
    if args.iteration == 'none':
        iteration = get_iteration(args=args)
    else:
        iteration = int(args.iteration)

    print(iteration)
    dirs['xml_save_dir'] = args.base_dir + '\\' + args.project + dirs['training_data_dir'] + str(iteration) + '\Predicted_XMLs\\'

    if iteration == 'none':
        print('ERROR: no trained models found \n\tplease use [--option train]')

    else:
        # check main directory exists
        make_folder(dirs['outDir'])
        make_folder(dirs['xml_save_dir'])

        # get all WSIs
        WSIs = []
        usable_ext=args.wsi_ext.split(',')

        for ext in usable_ext:
            WSIs.extend(glob(args.base_dir + '\\' + args.project + dirs['training_data_dir'] + str(iteration) + '\*' + ext))
        for wsi in WSIs:
            #try:
            predict_xml(args=args, dirs=dirs, wsi=wsi, iteration=iteration)
            #except KeyboardInterrupt:
            #    break
            #except:
            #print('!!! Prediction on ' + wsi + ' failed\nmoving on...')
        print('\n\n\033[92;5mPlease correct the xml annotations found in: \n\t' + dirs['xml_save_dir'])
        print('\nthen place them in: \n\t'+ args.base_dir + '\\' + args.project + dirs['training_data_dir'] + str(iteration) + '\\')
        print('\nand run [--option train]\033[0m\n')


def predict_xml(args, dirs, wsi, iteration):
    # reshape regions calc
    downsample = int(args.downsampleRateHR**.5)
    region_size = int(args.boxSizeHR*(downsample))
    step = int(region_size)-(50*2)

    # figure out the number of classes
    if args.classNum == 0:
        annotatedXMLs=glob(args.base_dir + '\\' + args.project + dirs['training_data_dir'] + str(iteration-1) + '\*.xml')
        classes = []
        for xml in annotatedXMLs:
            classes.append(get_num_classes(xml))
        classNum = max(classes)
    else:
        classNum = args.classNum

    if args.chop_data == 'True':
        # chop wsi
        fileID, test_num_steps = chop_suey(wsi, dirs, downsample, region_size, step, args)
        dirs['fileID'] = fileID
        print('Chop SUEY!\n')
    else:
        basename = os.path.splitext(wsi)[0]

        if wsi.split('.')[-1] != 'tif':
            slide=getWsi(wsi)
            # get image dimensions
            dim_x, dim_y=slide.dimensions
        else:
            im = Image.open(wsi)
            dim_x, dim_y=im.size

        fileID=basename.split('\\')
        dirs['fileID'] = fileID=fileID[len(fileID)-1]
        test_num_steps = file_len(dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + '_images' + ".txt")
    # call DeepLab for prediction
    print('Segmenting tissue ...\n')

    make_folder(dirs['outDir'] + fileID + dirs['img_save_dir'] + 'prediction')

    test_data_list = fileID + '_images' + '.txt'
    modeldir = args.base_dir + '\\' + args.project + dirs['modeldir'] + str(iteration) + '\HR'
    test_step = get_test_step(modeldir)
    print("\033[1;32;40m"+"starting prediction using model: \n\t" + modeldir + '\\' + str(test_step) + "\033[0;37;40m"+"\n\n")

    call(['python3', args.code_dir+'\Codes\Deeplab_network\main.py',
        '--option', 'predict',
        '--input_width',str(region_size),
        '--input_height',str(region_size),
        '--test_data_list', dirs['outDir']+fileID+dirs['txt_save_dir']+test_data_list,
        '--out_dir', dirs['outDir']+fileID+dirs['img_save_dir'],
        '--test_step', str(test_step),
        '--test_num_steps', str(test_num_steps),
        '--modeldir', modeldir,
        '--data_dir', dirs['outDir']+fileID+dirs['img_save_dir'],
        '--num_classes', str(classNum),
        '--gpu', str(args.gpu)])

    # un chop
    print('\nreconstructing wsi map ...\n')
    wsiMask = un_suey(dirs=dirs, args=args)
    '''
    plt.imshow(wsiMask)
    plt.show()
    exit()
    wsiMask[np.where(wsiMask==6)]=1
    '''
    # save hotspots
    if dirs['save_outputs'] == True:
        make_folder(dirs['outDir'] + fileID + dirs['mask_dir'])
        print('saving to: ' + dirs['outDir'] + fileID + dirs['mask_dir'] + fileID  + '.png')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(dirs['outDir'] + fileID + dirs['mask_dir'] + fileID + '.png', wsiMask)

    print('\n\nStarting XML construction: ')

    xml_suey(wsiMask=wsiMask, dirs=dirs, args=args, classNum=classNum, downsample=downsample)

    # clean up
    if dirs['save_outputs'] == False:
        print('cleaning up')
        rmtree(dirs['outDir']+fileID)


def get_iteration(args):
    currentmodels=os.listdir(args.base_dir + '\\' + args.project + '\MODELS\\')
    if not currentmodels:
        return 'none'
    else:
        currentmodels=list(map(int,currentmodels))
        Iteration=np.max(currentmodels)
        return Iteration

def get_test_step(modeldir):
    pretrains=glob(modeldir + '\*.ckpt*')
    maxmodel=0
    for modelfiles in pretrains:
        modelID=modelfiles.split('.')[-2].split('-')[1]
        try:
            modelID = int(modelID)
            if modelID>maxmodel:
                maxmodel=modelID
        except: pass
    return maxmodel

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory

def restart_line(): # for printing chopped image labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def getWsi(path): #imports a WSI
    OPENSLIDE_PATH = r"C:\Users\borgh\Downloads\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"

    if hasattr(os, 'add_dll_directory'):
        # Windows
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
    slide = openslide.OpenSlide(path)
    return slide

def file_len(fname): # get txt file length (number of lines)
    with open(fname) as f:
        for i, l in enumerate(f):
            pass

    if 'i' in locals():
        return i + 1

    else:
        return 0


def chop_suey(wsi, dirs, downsample, region_size, step, args): # chop wsi
    print('\nopening: ' + wsi)
    basename = os.path.splitext(wsi)[0]

    if wsi.split('.')[-1] != 'tif':
        slide=getWsi(wsi)
        # get image dimensions
        dim_x, dim_y=slide.dimensions
    else:
        im = Image.open(wsi)
        dim_x, dim_y=im.size

    fileID=basename.split('\\')
    dirs['fileID'] = fileID=fileID[len(fileID)-1]
    print('\nchopping ...\n')

    # make txt file
    print(dirs['outDir'] + fileID + dirs['txt_save_dir'])
    make_folder(dirs['outDir'] + fileID + dirs['txt_save_dir'])
    f_name = dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + ".txt"
    f2_name = dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + '_images' + ".txt"
    f = open(f_name, 'w')
    f2 = open(f2_name, 'w')
    f2.close()

    make_folder(dirs['outDir'] + fileID + dirs['img_save_dir'] + dirs['chopped_dir'])

    f.write('Image dimensions:\n')

    # make index for iters
    index_y=np.array(range(0,dim_y,step))
    index_x=np.array(range(0,dim_x,step))
    index_x[-1]=dim_x-step
    index_y[-1]=dim_y-step
    f.write('X dim: ' + str((index_x[-1]+region_size)/downsample) +'\n')
    f.write('Y dim: ' + str((index_y[-1]+region_size)/downsample) +'\n\n')
    f.write('Regions:\n')
    f.write('image:xStart:xStop:yStart:yStop\n\n')
    f.close()

    # get non white regions
    choppable_regions = get_choppable_regions(wsi=wsi, index_x=index_x, index_y=index_y, boxSize=region_size,white_percent=args.white_percent)

    print('saving region:')

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores, backend='threading')(delayed(chop_wsi)(yStart=i, xStart=j, idxx=idxx, idxy=idxy, f_name=f_name, f2_name=f2_name, dirs=dirs, downsample=downsample, region_size=region_size, args=args, wsi=wsi, choppable_regions=choppable_regions) for idxy, i in enumerate(index_y) for idxx, j in enumerate(index_x))

    test_num_steps = file_len(dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + '_images' + ".txt")
    print('\n\n' + str(test_num_steps) +' image regions chopped')

    return fileID, test_num_steps

def chop_wsi(yStart, xStart, idxx, idxy, f_name, f2_name, dirs, downsample, region_size, args, wsi, choppable_regions): # perform cutting in parallel
    if choppable_regions[idxy, idxx] != 0:
        yEnd = yStart+region_size
        #print(yEnd)
        xEnd = xStart+region_size
        #print(xEnd)
        xLen=xEnd-xStart
        yLen=yEnd-yStart

        if wsi.split('.') != 'tif':
            slide = getWsi(wsi)
            subsect= np.array(slide.read_region((xStart,yStart),0,(xLen,yLen)))
            subsect=subsect[:,:,:3]

        else:
            subsect_ = imread(wsi)[yStart:yEnd, xStart:xEnd, :3]
            subsect = np.zeros([region_size,region_size,3])
            subsect[0:subsect_.shape[0], 0:subsect_.shape[1], :] = subsect_

		#print(whiteRatio)
        imageIter = str(xStart)+str(yStart)

        f = open(f_name, 'a+')
        f2 = open(f2_name, 'a+')

        # append txt file
        f.write(imageIter + ':' + str(xStart/downsample) + ':' + str(xEnd/downsample)
            + ':' + str(yStart/downsample) + ':' + str(yEnd/downsample) + '\n')

		# resize images ans masks
        if downsample > 1:
            c=(subsect.shape)
            s1=int(c[0]/downsample)
            s2=int(c[1]/downsample)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                subsect=resize(subsect,(s1,s2), mode='constant')

        # save image
        directory = dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + dirs['chopped_dir']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(directory + dirs['fileID'] + str(imageIter) + args.imBoxExt,subsect)

        f2.write(dirs['chopped_dir'] + dirs['fileID'] + str(imageIter) + args.imBoxExt + '\n')
        f.close()
        f2.close()

        sys.stdout.write('   <'+str(xStart)+':'+str(xEnd)+' '+str(yStart)+':'+str(yEnd)+'>   ')
        sys.stdout.flush()
        restart_line()

def un_suey(dirs, args): # reconstruct wsi from predicted masks
    txtFile = dirs['fileID'] + '.txt'

    # read txt file
    f = open(dirs['outDir'] + dirs['fileID'] + dirs['txt_save_dir'] + txtFile, 'r')
    lines = f.readlines()
    f.close()
    lines = np.array(lines)

    # get wsi size
    xDim = np.uint32(float((lines[1].split(': ')[1]).split('\n')[0]))
    yDim = np.uint32(float((lines[2].split(': ')[1]).split('\n')[0]))
    print('xDim: ' + str(xDim))
    print('yDim: ' + str(yDim))

    # make wsi mask
    wsiMask = np.zeros([yDim, xDim]).astype(np.uint8)

    # read image regions
    for regionNum in range(7, np.size(lines)):
        # print regionNum
        sys.stdout.write('   <'+str(regionNum-7)+ ' of ' + str(np.size(lines)-8) +'>   ')
        sys.stdout.flush()
        restart_line()

        # get region
        region = lines[regionNum].split(':')
        region[4] = region[4].split('\n')[0]

        # read mask
        mask = imread(dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + 'prediction\\' + dirs['fileID'] + region[0] + '_smask.png')

        # get region bounds
        xStart = np.uint32(float(region[1]))
        #print('xStart: ' + str(xStart))
        xStop = np.uint32(float(region[2]))
        #print('xStop: ' + str(xStop))
        yStart = np.uint32(float(region[3]))
        if yStart < 0:
            yStart = 0
        #print('yStart: ' + str(yStart))
        yStop = np.uint32(float(region[4]))
        #print('yStop: ' + str(yStop))

        mask_part = wsiMask[yStart:yStop, xStart:xStop]
        ylen, xlen = np.shape(mask_part)
        mask = mask[:ylen, :xlen]

        # populate wsiMask with max
        #print(np.shape(wsiMask))
        wsiMask[yStart:yStop, xStart:xStop] = np.maximum(mask_part, mask).astype(np.uint8)
        #wsiMask[yStart:yStop, xStart:xStop] = np.ones([yStop-yStart, xStop-xStart])

    return wsiMask

def xml_suey(wsiMask, dirs, args, classNum, downsample):
    # make xml
    Annotations = xml_create()
    # add annotation
    for i in range(classNum)[1:]: # exclude background class
        Annotations = xml_add_annotation(Annotations=Annotations, annotationID=i)


    for value in np.unique(wsiMask)[1:]:
        # print output
        print('\t working on: annotationID ' + str(value))
        # get only 1 class binary mask
        binary_mask = np.zeros(np.shape(wsiMask)).astype('uint8')
        binary_mask[wsiMask == value] = 1

        # add mask to xml
        pointsList = get_contour_points(binary_mask, args=args, downsample=downsample,value=value)
        for i in range(np.shape(pointsList)[0]):
            pointList = pointsList[i]
            Annotations = xml_add_region(Annotations=Annotations, pointList=pointList, annotationID=value)

    # save xml
    xml_save(Annotations=Annotations, filename=dirs['xml_save_dir']+'\\'+dirs['fileID']+'.xml')

def get_contour_points(mask, args, downsample,value, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    pointsList = []

    for j in range(np.shape(maskPoints)[0]):
        if cv2.contourArea(maskPoints[j]) > args.min_size[value-1]:
            pointList = []
            for i in range(0,np.shape(maskPoints[j])[0],4):
                point = {'X': (maskPoints[j][i][0][0] * downsample) + offset['X'], 'Y': (maskPoints[j][i][0][1] * downsample) + offset['Y']}
                pointList.append(point)
            pointsList.append(pointList)
    return pointsList

### functions for building an xml tree of annotations ###
def xml_create(): # create new xml tree
    # create new xml Tree - Annotations
    Annotations = ET.Element('Annotations')
    return Annotations

def xml_add_annotation(Annotations, annotationID=None): # add new annotation
    # add new Annotation to Annotations
    # defualts to new annotationID
    if annotationID == None: # not specified
        annotationID = len(Annotations.findall('Annotation')) + 1
    Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': '4', 'Visible': '1', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': str(xml_color[annotationID-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
    Regions = ET.SubElement(Annotation, 'Regions')
    return Annotations

def xml_add_region(Annotations, pointList, annotationID=-1, regionID=None): # add new region to annotation
    # add new Region to Annotation
    # defualts to last annotationID and new regionID
    Annotation = Annotations.find("Annotation[@Id='" + str(annotationID) + "']")
    Regions = Annotation.find('Regions')
    if regionID == None: # not specified
        regionID = len(Regions.findall('Region')) + 1
    Region = ET.SubElement(Regions, 'Region', attrib={'NegativeROA': '0', 'ImageFocus': '-1', 'DisplayId': '1', 'InputRegionId': '0', 'Analyze': '0', 'Type': '0', 'Id': str(regionID)})
    Vertices = ET.SubElement(Region, 'Vertices')
    for point in pointList: # add new Vertex
        ET.SubElement(Vertices, 'Vertex', attrib={'X': str(point['X']), 'Y': str(point['Y']), 'Z': '0'})
    # add connecting point
    ET.SubElement(Vertices, 'Vertex', attrib={'X': str(pointList[0]['X']), 'Y': str(pointList[0]['Y']), 'Z': '0'})
    return Annotations

def xml_save(Annotations, filename):
    xml_data = ET.tostring(Annotations, pretty_print=True)
    #xml_data = Annotations.toprettyxml()
    f = open(filename, 'wb')
    f.write(xml_data)
    f.close()

def read_xml(filename):
    # import xml file
    tree = ET.parse(filename)
    root = tree.getroot()
