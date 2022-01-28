import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
import scipy.sparse
from tqdm.auto import tqdm 
import time
import pickle
import os.path


from index_class import cluster_index 
from index_class import data_io

class IVFIndex:
    def __init__(self, num_cluster_indexes, data_name, metric='angular', **kwargs):
        self.metric = metric
        self.num_cluster_indexes = num_cluster_indexes 
        self.data_name = data_name #Stores the name of the dataset for saving, and file I/O.
        #These two values determine the file I/O for the upper level indexing.
        
        self.cluster_map = None
        self.cluster_centers = None
        self.n = None 
        self.ivf_map = dict([]) # Dict storing ivf_map[<cluster id>] -> array([<data ids>]) 
        self.ivf_data_res = dict([]) #Dict storing ivf_data_res[<cluster id>] -> array(<residuals of data>) 
        
    def __str__(self):
        return "IVFIndex_%s_%d" % (self.data_name, self.num_cluster_indexes)
    
    def str_cluster_indexing(self, c_index_class, *init_args, **buildkwargs):
        return ("%s_%d_%s" %(self.data_name, self.num_cluster_indexes, c_index_class(*init_args).str_name(**buildkwargs)))
    
    def filename_ivf_index(self): 
        return "results/IVFIndex_%s_%d" % (self.data_name, self.num_cluster_indexes)
    
    def filename_cluster_indexes(self, c_index_class, *init_args, **buildkwargs): 
        return ("results/ClusterIndexes_%s_%d_%s" % 
                (self.data_name, self.num_cluster_indexes, c_index_class(*init_args).str_name(**buildkwargs)))
                
    
    def compute_ivf_map(cluster_map):
        c_order = np.argsort(cluster_map)
        c_sorted = cluster_map[c_order]
        c_ix_partition = (np.diff(c_sorted, prepend=c_sorted[0]) != 0).nonzero()[0]
        ivf_map = {}
        for i in range(len(c_ix_partition) + 1): 
            i_p = 0 if i == 0 else c_ix_partition[i-1]
            i_n = len(c_order) if i == len(c_ix_partition) else c_ix_partition[i]
            ivf_map[i] = np.sort(c_order[i_p:i_n]) #Sorted so as to allow for indexing into data
        return ivf_map 

    
    def build_ivf_index(self, X, save_index=True, **kwargs): #Gets a hd5 file X. 
        fname = self.filename_ivf_index()
        if os.path.isfile(fname):
            ivfix = data_io.load_pickle(fname)
            self.n = ivfix.n
            self.cluster_map = ivfix.cluster_map
            self.cluster_centers = ivfix.cluster_centers
            self.ivf_map = ivfix.ivf_map
            self.ivf_data_res = ivfix.ivf_data_res
            print("IVF Indexing loaded from file: %s" %fname)
        else:
            self.n = X.shape[0]

            print("Computing IVFIndex for %s" % str(self))
            t1 = time.time()

            kmeans = KMeans(n_clusters=self.num_cluster_indexes, random_state=0, 
                            copy_x=True, n_init=1, verbose=1).fit(X)

            self.cluster_map = kmeans.labels_
            self.cluster_centers = kmeans.cluster_centers_
            self.ivf_map = IVFIndex.compute_ivf_map(self.cluster_map)

            for i in tqdm(range(self.num_cluster_indexes)): #Iterate over every cluster and build index.
                self.ivf_data_res[i] = X[self.ivf_map[i]] - self.cluster_centers[i] #Compute the residuals 

            #Save the upper-level index (full instance)
            if save_index == True:
                data_io.save_pickle(self.filename_ivf_index(), self)

            t2 = time.time()
            print("IVFIndex completed in time: %3.3f s" % (t2 - t1))

    
    
    def build_cluster_indexes(self, cluster_index_class, *indexinitargs, save_index=True, **indexbuildkwargs):
        fname = self.filename_cluster_indexes(cluster_index_class, *indexinitargs, **indexbuildkwargs)
        cluster_indexes = {}
        if os.path.isfile(fname):
            #Load the values of the indexes from the file.
            old_cluster_indexes = data_io.load_pickle(fname)
            cluster_indexes = {i : cluster_index_class(*indexinitargs) for i in range(self.num_cluster_indexes)}
            for i in range(len(cluster_indexes)):
                #Note that init args are already set by the fact that we initialized cluster_indexes with indexinitargs
                #we only need to set the args set by build_index, i.e. assignments, centers, ids, lengths. 
                cluster_indexes[i].assignments = old_cluster_indexes[i].assignments
                cluster_indexes[i].centers = old_cluster_indexes[i].centers
                cluster_indexes[i].ids = old_cluster_indexes[i].ids
                cluster_indexes[i].lengths = old_cluster_indexes[i].lengths
            
            print("Cluster indexes loaded from file: %s" % fname)
        
        else: 
            cluster_indexes = {i : cluster_index_class(*indexinitargs) for i in range(self.num_cluster_indexes)}
            
            print("Computing cluster indexing for %s" % 
                  self.str_cluster_indexing(cluster_index_class, *indexinitargs, **indexbuildkwargs))
            t1 = time.time()

            #Iterate through clusters and build the indexes by calling buld method of each instance.
            for i in tqdm(range(self.num_cluster_indexes)):
                cluster_indexes[i].build_index(self.ivf_data_res[i], self.ivf_map[i], **indexbuildkwargs)
            
            t2 = time.time()
            print("Cluster indexing completed in time: %3.3f s" % (t2 - t1))
            
            if save_index == True: 
                data_io.save_pickle(fname, cluster_indexes)
            
        return cluster_indexes
    
    #For now, only implementing inner-product (angular) distances.
    def query(self, X, q, n_clusters, n_final, n_per_cluster, cluster_indexes, rerank_exact=True): 
        #initialize the distances and ids arrays to inf and nans. 
        #To be filled by cluster index query method.
        all_dists = np.full(n_per_cluster*n_clusters, np.inf)
        all_ids = np.full(n_per_cluster*n_clusters, np.nan, dtype=np.intc)
        
        #Compute the closes n_clusters centers to search
        cluster_ids = np.argpartition(self.cluster_centers@(-q), n_clusters-1)[:n_clusters]
        
        for i in range(len(cluster_ids)):
            curr_c_id = cluster_ids[i]
            if self.metric == 'angular': 
                dists = cluster_indexes[curr_c_id].query_angular_dists(q)
            elif self.metric == 'euclidean':
                dists = cluster_indexes[curr_c_id].query_euclidean_dists(q)
            
            n_temp = min(len(dists), n_per_cluster) #number of ids returned by the cluster
            top_n_order = np.argpartition(dists, n_temp-1)[:n_temp]

            c_ids = cluster_indexes[curr_c_id].ids[top_n_order]
            c_dists = np.zeros(n_temp)
            if rerank_exact == True: #Compute the exact scores 
                c_ids_sort_order = np.argsort(c_ids) 
                c_dists[c_ids_sort_order] = X[c_ids[c_ids_sort_order]]@(-q) #Output in same order as c_ids
            else: #Comptue the approximate scores
                #-<x, q> = -<x-c, q> - <c, q>
                c_dists = dists[top_n_order] - (self.cluster_centers[curr_c_id]@q)
            
            curr_i = i*n_per_cluster 
            all_ids[curr_i : (curr_i + n_temp)] = c_ids
            all_dists[curr_i : (curr_i + n_temp)] = c_dists
        return all_ids[np.argpartition(all_dists, n_final-1)[:n_final]]
            
            
        
        
            
                
            
        
        


