#!/usr/bin/python
import glob
import numpy as np

# filePath = "EMG_data_for_gestures-master/*/*.txt"

def readFiles(filePath):
  #read files
  fileList = glob.glob(filePath)
  data = list()

  for fileName in fileList:
    data.append(readFile(fileName))

  return data,fileList


def readFile(fileName):
  file = open(fileName, "r")
  lines = file.readlines()

  rowSize = len(lines)-1
  colSize = len(lines[0].split())
  array = np.zeros((rowSize,colSize))
  
  countLine = 0
  for line in lines[1:]:
    countCol = 0
    for col in line.split():
      array[countLine, countCol] = float(col)
      countCol += 1
    countLine += 1
  
  return array