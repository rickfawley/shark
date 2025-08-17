# ================================================================ #
# DESCRIPTION
# ================================================================ #
# This code evaluates the SHARK Algorithm against contemporaries.
# This implementation leverages additive decomposability
#
#
#
#

# ---------------------------------------------------------------- #
# IMPORT LIBRARIES
# ---------------------------------------------------------------- #

import random
import os
import concurrent.futures

import numpy          as np
import sklearn        as sk
import unsupervisedpy as up

from   sklearn        import metrics
from   typing         import Literal
from   datetime       import datetime  

# ================================================================ #
# Environmental variables and parameters
# ================================================================ #

experimentID     : str = r'Data5000x50_50k'
noiseFeatures    : int = 25

myDataFolder     : str = r'../Data/'

reportAccuracyDP : int = 3
iterationCount   : int = 25

algorithms   = Literal[
                    'SHARK', 
                    'LWK-Means',
                ]

# ================================================================ #
# Functions and Lambda
# ================================================================ #

# ---------------------------------------------------------------- #
# Logging helper
# ---------------------------------------------------------------- #

def log(
        logMessage,
        logFilename,
    ):

    print(logMessage)

    with open(logFilename, "a") as f:
        f.write(f"{logMessage}\n")
    #end-with

    return

# ---------------------------------------------------------------- #
# SHARK Algorithm
# ---------------------------------------------------------------- #

def SHARK(
        data : np.ndarray, 
        k    : int,
    ):

    # ---------------------------------------------------------------- #
    # Initialise weights, all equal
    # ---------------------------------------------------------------- #

    weights = np.full((data.shape[1]), 1/data.shape[1])

    # ---------------------------------------------------------------- #
    # Initialise centroids, ensuring each centroid is different
    # ---------------------------------------------------------------- #

    uniqueDataItems = np.unique(data, axis=0)
    centroids       = uniqueDataItems[random.sample(range(uniqueDataItems.shape[0]), k),:]
   
    # ---------------------------------------------------------------- #
    # initialising labels from the previous iteration (to be compared 
    # with the current labels)
    # ---------------------------------------------------------------- #

    previousLabels = np.zeros((data.shape[0]))
   
    while True:

        # ---------------------------------------------------------------- #
        # Assign each data point to the neares cluster.
        # distances has shape (n, k) so that distances[i,j] has the 
        # distance between row i and centroid j
        # ---------------------------------------------------------------- #

        distances = np.zeros((data.shape[0], k))

        for k_i in range(k):
            distances[:,k_i] = (weights * (data - centroids[k_i])**2).sum(axis=1)
        #end-for

        # ---------------------------------------------------------------- #
        # Put inf instead of nan (in case of empty cluster)        
        # ---------------------------------------------------------------- #

        labels = np.where(np.isnan(distances), np.inf, distances).argmin(axis=1)

        # ---------------------------------------------------------------- #
        # If there is no change in the clustering, then stop
        # ---------------------------------------------------------------- #

        if np.all(previousLabels==labels):
            break
        #end-if

        # ---------------------------------------------------------------- #
        # Update centroids
        # ---------------------------------------------------------------- #

        for k_i in range(k):
            centroids[k_i] = data[labels==k_i].mean(axis=0)
        #end-for
       
        # ---------------------------------------------------------------- #
        # Update weights
        # Initialise phi to have shape (m,) where m is the number of features
        # ---------------------------------------------------------------- #

        phi = np.zeros((data.shape[1]))

        # ---------------------------------------------------------------- #
        # Compute exact Shapley values via within-cluster dispersion
        # (Matches Theorem 1 in the paper)
        # ---------------------------------------------------------------- #

        for k_i in range(k):
            
            clusterPoints = data[labels == k_i]
            
            if clusterPoints.shape[0] > 0:
                diff = clusterPoints - centroids[k_i]
                phi += (diff ** 2).sum(axis=0)
            #end-if

        #end-for

        # ---------------------------------------------------------------- #
        # Calculate weights
        # ---------------------------------------------------------------- #

        with np.errstate(divide='ignore'):
            weights = np.where(phi > 0, 1.0 / phi, 0.0)
        #end-with
        
        weights /= weights.sum()

        previousLabels = labels

    return labels, weights, distances
               

# ---------------------------------------------------------------- #
# Function to calculate the kmeans criterion
# ---------------------------------------------------------------- #

def getKmeansScore(
        data      : np.ndarray, 
        centroids : np.ndarray, 
        labels    : np.ndarray, 
        k         : int,
    ):

    score = 0

    for k_i in range(k):
        score += ((data[labels==k_i] - centroids[k_i]) **2 ).sum()
    #end-for

    return score

# ---------------------------------------------------------------- #
# Function to load data from experiment data folder
# ---------------------------------------------------------------- #

def loadData(
        experimentID : str, 
        id           : int, 
        NF           : int = 0,
    ):

    print(f"Experiment: {experimentID}")

    if NF == 0:
        dataFileName = f"{experimentID}_{str(id)}.csv"
    else:
        dataFileName = f"{experimentID}_{str(NF)}NF_{str(id)}.csv"
    #end-if

    labelsFileName = f"{experimentID}_{str(id)}_Labels.csv"

    dataFile   = f"{myDataFolder}{experimentID}/{dataFileName}"
    labelsFile = f"{myDataFolder}{experimentID}/{labelsFileName}"

    print(f"{dataFile}")
    
    data = np.loadtxt(dataFile,   delimiter=',')
    y    = np.loadtxt(labelsFile, delimiter=',')

    print(f"Number of Observations:   {data.shape[0]}\t(inferred from data)")
    print(f"Number of Features:       {data.shape[1]}\t(inferred from data)")
    print(f"Number of Clusters:       {np.unique(y).size}\t(inferred from data)")
    print(f"Number of Noise Features: {NF}\t(supplied)")
    print(f"\n")

    return data, y

# ---------------------------------------------------------------- #
# Function to process a single dataset using SHARK
# ---------------------------------------------------------------- #

def run_SHARK_singleDataset(
        data, 
        k,
        iterations : int = 25,
    ):

    minimumInertia = float('inf')
    bestLabels     = np.array([])

    for _ in range(iterations):
         
        labels, weights, distances = SHARK(data, k)

        if np.unique(labels).size == k:

            sumOfDistances = distances.min(axis=1).sum()

            if sumOfDistances < minimumInertia:
                minimumInertia = sumOfDistances
                bestLabels     = labels
            #end-if

        #end-if

    #end-for

    return bestLabels

# ---------------------------------------------------------------- #
# Function to process a single dataset using lasso weighted kmeans
# ---------------------------------------------------------------- #

def run_Lwkmeans_singleDataset(
        data, 
        k,
        iterations : int = 25,
    ):
    
    minimumInertia = float('inf')
    bestLabels     = np.array([])

    lambda_ = None

    for _ in range(iterations):

        try:
            lwk = up.cluster.LWKmeans(k, lambda_).fit(data)
        except:
            continue
        #end-try

        lambda_ = lwk.lambd

        if np.unique(lwk.labels).size == k:

            sumOfDistances = lwk.distances.sum()

            if sumOfDistances < minimumInertia:
                minimumInertia = sumOfDistances
                bestLabels     = lwk.labels
            #end-if

        #end-if

    #end-for

    return bestLabels

# ---------------------------------------------------------------- #
# Execute an experiment series using single threading
# ---------------------------------------------------------------- #

def runExperiments_singleThreading(
        experimentID : str, 
        NF           : int, 
        algorithm    : algorithms,
        iterations   : int = 25,
    ):

    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logFilename = f"{algorithm}_{experimentID}_{timestamp}.txt"
    
    log(f"Experiment started: {timestamp}",logFilename)
    log(f"File core:          {experimentID}",logFilename)
    log(f"Noise features:     {NF}",logFilename)
    log(f"Algorithm:          {algorithm}",logFilename)

    results = []
    
    for dt in range(1,51):

        print(dt)
        data, y = loadData(experimentID, dt, NF)
        k       = np.unique(y).size

        if algorithm == 'SHARK':
            labels = run_SHARK_singleDataset(data, k, iterations)
        elif algorithm == 'LWK-Means':
            labels = run_Lwkmeans_singleDataset(data, k, iterations)
        #end-if

        if labels.size != 0:
            results.append(metrics.adjusted_rand_score(labels,y))
        #end-if

    #end-for

    r = np.array(results)

    meanARI = np.round(r.mean(), reportAccuracyDP)
    stdARI  = np.round(r.std(),  reportAccuracyDP)

    log("Finished all datasets\n",logFilename)

    log(f"Results count:     {r.size}",logFilename)
    log(f"Mean ARI:          {meanARI}",logFilename)
    log(f"Std  ARI:          {stdARI}",logFilename)

    log(f"Experiment ended:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",logFilename)
    log(f"Results written to {logFilename}",logFilename)

    print(f"File:            {str(experimentID)}")
    print(f"NF:              {str(NF)}")

    return meanARI, stdARI

# ---------------------------------------------------------------- #
# Process one file using parallel processing
# ---------------------------------------------------------------- #

def runSingleDataset_parallelProcessing(
        experimentID : str, 
        NF           : int, 
        algorithm    : algorithms, 
        dt,
    ):
    
        print(f"Processing dataset {dt} {str(datetime.now()).split('.')[0]}", flush=True)
    
        try:

            data, y = loadData(experimentID, dt, NF)
            k = np.unique(y).size

            if algorithm == 'SHARK':
                labels = run_SHARK_singleDataset(data, k, iterations=iterationCount)
            elif algorithm == 'LWK-Means':
                labels = run_Lwkmeans_singleDataset(data, k, iterations=iterationCount)
            #end-if

            if labels.size != 0:
                return sk.metrics.adjusted_rand_score(labels, y)
            #end-if

        except Exception as e:
            print(f"Error on dataset {dt}: {e}", flush=True)
        #end-try

        return None

# ---------------------------------------------------------------- #
# Execute an experiment series using parallel processing
# ---------------------------------------------------------------- #

def runExperiments_parallelProcessing(
        experimentID : str, 
        NF           : int, 
        algorithm    : algorithms,
    ):

    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logFilename = f"{algorithm}_{experimentID}_{timestamp}.txt"
    
    log(f"Experiment started: {timestamp}",logFilename)
    log(f"File core:          {experimentID}",logFilename)
    log(f"Noise features:     {NF}",logFilename)
    log(f"Algorithm:          {algorithm}",logFilename)

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:

        futures = {
            executor.submit(
                runSingleDataset_parallelProcessing,
                experimentID, 
                NF, 
                algorithm, 
                dt,
            ) : dt for dt in range(1, 51)
        }

        results = []
        
        for future in concurrent.futures.as_completed(futures):

            result = future.result()

            if result is not None:
                results.append(result)
            #end-if

        #end-for

    #end-with

    r = np.array(results)

    meanARI = np.round(r.mean(), reportAccuracyDP)
    stdARI  = np.round(r.std(), reportAccuracyDP)

    log("Finished all datasets\n",logFilename)
    log(f"Results count:     {r.size}",logFilename)
    log(f"Mean ARI:          {meanARI}",logFilename)
    log(f"Std  ARI:          {stdARI}",logFilename)

    log(f"Experiment ended:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",logFilename)
    log(f"Results written to {logFilename}",logFilename)

    print(f"File:            {str(experimentID)}")
    print(f"NF:              {str(NF)}")

    return meanARI, stdARI

# ================================================================ #
# MAIN PROCESSING
# ================================================================ #

if __name__ == "__main__":

    startTime = datetime.now()

    print(f'Starting {startTime.strftime("%Y-%m-%d %H:%M:%S")}\n')

    # ---------------------------------------------------------------- #
    # Examples
    # ---------------------------------------------------------------- #

    # Run the experiment on a configuration with no noise features
    #     meanARI, stdARI = runExperiments_singleThreading('Data5000x50_10k',0, 'SHARK')

    # Run the same configuration but with noise features. The 25 here is the quantity of noise features

    #     meanARI, stdARI = runExperiments_singleThreading('Data5000x50_10k', 25, 'SHARK')
    #     meanARI, stdARI = runExperiments_parallelProcessing('Data5000x50_10k',0, 'LWK-Means')

    # ---------------------------------------------------------------- #
    # Payload
    # ---------------------------------------------------------------- #

    meanARI, stdARI = runExperiments_parallelProcessing(experimentID, noiseFeatures, 'SHARK')

    endTime  = datetime.now()
    duration = endTime - startTime

    print(f'\nEnded    {endTime.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Duration {str(duration).split(".")[0]} ({duration.total_seconds():.2f})')

#end-main-processing

