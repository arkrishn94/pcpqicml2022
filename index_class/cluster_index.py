import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import LogisticRegression
import scipy.sparse
from scipy.optimize import minimize
from scipy import integrate

import time

from tqdm.auto import tqdm 

'''Highest level abstraction of a cluster index. Goal is to compute a an A*B that approximates
the data X. The Class implements the highest-level functions for this kind of index -- i.e. init, train A abtraction, train B abstraction, query, str etc. ClusterIndex needs build_method as a function that when build(X, train_set) is called will run on X to return A and B matrices respectively; optionally if there is a training set then train_set can be used. 

<ClusterIndex>.assignments is a sparse.csr_matrix type to reduce memory.
<ClusterIndex>.centers is a dense matrix '''
class ClusterIndex: 
    def __init__(self, metric='angular'): 
        if metric not in ('angular', 'euclidean'):
            raise NotImplementedError("ClusterIndex doesn't support metric %s" % metric)
        self.metric = metric
        self.assignments = None #This is the A matrix in the AB formulation. Stored in sparse.csr_format
        self.centers = None #This is the B matrix stored as a regular numpy array.
        self.ids = None #This is a dict storing the ids ofr each of the vectors in the index.
        self.lengths = None
    
    '''Main method to build the index. Calls subclasses fit method to obtain 
    assignments and centers. Additionally all processing of index, X etc. is done 
    here. Ids are set according to input. <train_set>, deafult is None, is 
    used if training is needed to build index.'''
    def build_index(self, X, ids, **kwargs): 
        if X.dtype != np.float32: X = X.astype(np.float32) #Quantize input to 4byte floats.
        self.ids = ids
        self.assignments, self.centers = self.fit(X, **kwargs) 


    '''Query returns the top n entries based on self.metric 
    using the A*B quantizing formulation. Uses query_<metric>_dists 
    to obtain dists for all points in X. query_<metric>_dists is implemented
    in each subclass'''
    def query(self, q, n): 
        if self.metric == 'euclidean': dists = self.query_euclidean_dists(q)
        elif self.metric == 'angular': dists = self.query_angular_dists(q)
        #output the ids from the top n sorted distances.
        return self.ids[np.argpartition(dists, n-1)[:n]] 
    
    '''Generic functions that compute distances if assignments and centers are matrices. 
    Can be overridden by subclass specific functions.'''
    def query_angular_dists(self, q): 
        # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)
        return self.assignments.dot(self.centers.dot(-q)) # Use A.dot(B) for np.dot(A, B)

    def query_euclidean_dists(self, q): #################WRONG!!!! lengths --> ||A.dot(B)||
        # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab  # noqa
        return self.lengths - 2*self.assignments.dot(self.centers.dot(q))
    
    
    
    '''Computes the relative reconstruction error for each point in X using assignmetns
    and centers.'''
    def compute_reconstruction_error(self, X): return ((X - self.assignments.dot(self.centers))**2).sum(-1)
    
    #Use only if needed
    def compute_approx_points(self): return self.assignments.dot(self.centers)
       
    '''Functions for computing sparsity of assignment matrices'''
    #Returns avg sparsity of the matrix A
    def assignments_sparsity(self): 
        N = self.assignments.shape[0]
        M = self.assignments.shape[1]
        return 1.- self.assignments.count_nonzero()/(N*M)
    
    #Returns the sparsities of each row in A matrix
    def assignments_row_sparsities(self):
        s = np.zeros(self.assignments.shape[0])
        N = self.assignments.shape[0]
        M = self.assignments.shape[1]
        for r in range(N): s[r] = 1. - self.assignments[r].count_nonzero()/M
        return s
    
    def assignments_lengths(self): return scipy.sparse.linalg.norm(self.assignments, ord=2, axis=1)
    



    
'''sub_index_inst needs to be initialized instance of sub-index class.'''
class PQIndex(ClusterIndex): 
    def __init__(self, k, M, sub_index_inst, metric='angular'):
        ClusterIndex.__init__(self, metric)
        self.k = k
        self.M = M 
        self.sub_index_inst = sub_index_inst
    
    def str_name(self, **kwargs): 
        return ("PQ(%s)_M=%d" %(self.sub_index_inst.str_name(**kwargs), self.M))
    
    
    '''Quantizes each row to a single scaled signed value. 
    Modifies input assignments in place.'''
    def quantize_each_row(self, labels):
        for r in (range(labels.shape[0])):
            (rids, cids) = np.nonzero(labels[r])
            data = (labels[r, cids]).toarray()
            quant_data = np.sign(data)*np.mean(np.abs(data))
            labels[r, cids] = quant_data
        
    '''Assumes that each row is already quantized to a single value.'''
    def quantize_across_rows(self, labels, q_val): 
        (rids, cids) = np.nonzero(labels)
        vals = np.abs((labels[rids, cids]).A1)
        val_map, val_c = KMeansIndex.kmeans_mat(vals, q_val)
        labels[rids, cids] = (val_c[val_map])*np.sign((labels[rids, cids]).A1)
    
    def fit(self, X, **kwargs): 
        k, M = self.k, self.M
        col_ind_grid = np.zeros(M+1).astype(np.intc)
        col_ind_grid[:M] = np.arange(0, X.shape[1], X.shape[1]//self.M)[:M]
        col_ind_grid[M] = X.shape[1]
        
#         B = np.zeros((k*M, X.shape[1]))
        #Initiating A and B as a lil because it supports flexible slicing and changes to sparsity structure
        #In addition, A, B can be stored with low mem overhead in PQ type indexes.
        B = scipy.sparse.lil_matrix((k*M, X.shape[1]), dtype=np.float32)
        A = scipy.sparse.lil_matrix((X.shape[0], k*M), dtype=np.float32) 
        for m in range(M): 
            #Compute the lower and upper indices for the bucket
            l_m, u_m = col_ind_grid[m], col_ind_grid[m+1]
            
            sub_buildkwargs = kwargs.copy() #Build kewword args for subindex. Slice train_set if valid.
            if 'train_set' in sub_buildkwargs: 
                sub_buildkwargs['train_set'] = kwargs['train_set'][:, l_m:u_m]
                            
            A_temp, B_temp = self.sub_index_inst.fit(X[:, l_m:u_m], **sub_buildkwargs)#Build index for chunk of data 
            A[:, m*k:(m+1)*k] = A_temp
            B[m*k:(m+1)*k, l_m:u_m] = B_temp
        
        labels = A.tocsr()
        
        if ('SQ' in kwargs): #Scalar quanitze the projections 
            SQ = kwargs['SQ']
            self.quantize_each_row(labels) #Quantize the rows down to one scalar value.
            if type(SQ) == int: 
                self.quantize_across_rows(labels, SQ) #quantize the projections to SQ values.
        
        return (labels, B)
    
    
    
    
    
'''KMeans Index subclass of ClusterIndex. Takes number of clusters, k and 
builds an index. self.assignments are an array of indices of clusters and self.centers 
are a k x d array corresponding to the cluster centers. '''
class KMeansIndex(ClusterIndex):
    def __init__(self, k, metric='angular'): 
        ClusterIndex.__init__(self, metric)
        self.k = k
    
    def str_name(self, **kwargs): 
        return ("KMeans_k=%d" %self.k)
    
    '''Method that can be called from outside the instance.
    Used for running k-means on X.'''
    def kmeans_mat(X, k): 
        #If k is larger than data then return the data as the centers. With some dummy centers.
        if (k >= X.shape[0]): 
            if len(X.shape) > 1: 
                centers = np.zeros((k, X.shape[1]))
            else: 
                centers = np.zeros(k)
            centers[:X.shape[0]] = X
            return (np.arange(X.shape[0]), centers)
        else: 
            if len(X.shape) == 1:
                km = KMeans(n_clusters=k, random_state=0, n_init=1).fit(X.reshape(-1, 1))
                labels, centers = km.labels_, km.cluster_centers_.ravel()
            else: 
                km = KMeans(n_clusters=k, random_state=0, n_init=1).fit(X)
                labels, centers = km.labels_, km.cluster_centers_
            return (labels, centers)
    
    def fit(self, X, **kwargs): 
        km_labels, km_centers = KMeansIndex.kmeans_mat(X, self.k)
        labels = scipy.sparse.csr_matrix( (np.ones(X.shape[0]), (np.arange(X.shape[0]), km_labels)),shape=(X.shape[0], self.k))
        return (labels, km_centers)

    
    
    

class ScannIndex(ClusterIndex):
    atol_ip = 1e-3
    rel_tol = 1e-4
    def __init__(self, k, metric='angular', T=0.2): 
        ClusterIndex.__init__(self, metric)
        self.k = k
        self.T = T
        self.atol_ip = 1e-3
    
    def str_name(self, **kwargs): 
        if 'project' in kwargs:
            if 'SQ' in kwargs: 
                return ("Scann_k=%d_project_SQ=%s" %(self.k, str(kwargs['SQ'])))
            else: 
                return ("Scann_k=%d_project" %(self.k))
        else: 
            return ("Scann_k=%d" %self.k)
    
    def compute_h_vals(X, T, delt=1e-2, atolip=1e-3): 
        norms = np.linalg.norm(X, ord=2, axis=1)
        
        def f_h_para(x): return (np.sin(x)**(X.shape[1]-2) - np.sin(x)**(X.shape[1]))
        def f_h_orth(x): return (np.sin(x)**(X.shape[1]))
        
        x_grid = np.arange(0, np.pi + np.pi*delt, np.pi*delt)
                
        para_cum_int = scipy.integrate.cumtrapz(f_h_para(x_grid), x_grid, initial=0.)
        orth_cum_int = scipy.integrate.cumtrapz(f_h_orth(x_grid), x_grid, initial=0.)
        
        h_para_vals = np.zeros(X.shape[0])
        h_orth_vals = np.zeros(X.shape[0])
        for i in range(X.shape[0]): 
            if norms[i] > T + atolip: 
                grid_ix = np.intc(np.rint(np.arccos(T/norms[i])/(np.pi*delt)))    
                h_para_vals[i] = (X.shape[1]-1)*para_cum_int[grid_ix]
                h_orth_vals[i] = orth_cum_int[grid_ix]
            else: 
                h_para_vals[i] = 0.
                h_orth_vals[i] = 0.
        return (h_para_vals, h_orth_vals)
        
    
    def compute_new_center(cluster, cluster_n, h_para_vals, h_orth_vals):
        C_n = cluster_n
        C_n_hscaled = C_n*(np.sqrt(h_para_vals - h_orth_vals)[:, np.newaxis])
        A = (C_n_hscaled.T)@C_n_hscaled
        np.fill_diagonal(A, A.diagonal() + h_orth_vals.sum())
        return np.linalg.inv(A).dot((cluster*(h_para_vals[:, np.newaxis])).sum(axis=0))
        
    def update_centers_map(self, data, centers, center_map):
        dots = data@centers.T
        center_map[:] = np.argmin(-2*(dots) + (centers**2).sum(axis=1), axis=1)

    
    def update_centers(self, data, data_n, h_para_vals, h_orth_vals, centers, center_map):
        (k,d) = centers.shape
        for i in range(k):
            cluster_filter = center_map==i
            cluster = data[cluster_filter]
            if len(cluster) == 0:
                center = data[np.random.randint(data.shape[0])]
            else:
                norms = np.linalg.norm(cluster,ord=2,axis=1)
                if np.all(norms <= (self.T + ScannIndex.atol_ip)):
                    center = cluster[np.random.randint(cluster.shape[0])]
                else:
                    center = ScannIndex.compute_new_center(cluster, data_n[cluster_filter],
                                                       h_para_vals[cluster_filter], 
                                                       h_orth_vals[cluster_filter])
            centers[i,:] = center
    
    
    def compute_alpha(x, x_tilde, x_h_p, x_h_o):
        if (x_h_p == 0. or x_h_o == 0.): 
            alpha = 1.
        else:
            alpha = (2*x_h_p*x_tilde.dot(x))/(2*((x_h_p - x_h_o)*(x_tilde.dot(x)**2)/(x**2).sum() + x_h_o*((x_tilde**2).sum()))) 
        return alpha
    
    def compute_projections(self, X, labels, centers, h_p, h_o): 
        for r in (range(labels.shape[0])):
            (rids, cids) = np.nonzero(labels[r])
            x_tilde = (centers[cids].sum(axis=0))
            alpha = ScannIndex.compute_alpha(X[r], x_tilde, h_p[r], h_o[r])
            labels[r, cids] = alpha

                    
    
    def fit(self, X, **kwargs):
        
        #If k is larger than the |X| then initialize remainnig cetners to 0.
        if (self.k >= X.shape[0]): #Just return the labels and centers if k < |X|
            if len(X.shape) > 1: 
                centers = np.zeros((self.k, X.shape[1]))
            else: 
                centers = np.zeros(self.k)
            centers[:X.shape[0]] = X
            center_map = np.zeros(X.shape[0], dtype=np.intc)
            self.update_centers_map(X, centers, center_map)
            labels = scipy.sparse.csr_matrix((np.ones(X.shape[0]), (np.arange(X.shape[0]), center_map)),
                                         shape=(X.shape[0], self.k))
        else:         
            ixs = np.random.choice(X.shape[0], size=self.k, replace=False)
            centers = X[np.sort(ixs)] #CHoose random k centers from the data

            center_map = np.zeros(X.shape[0], dtype=np.intc)
            self.update_centers_map(X, centers, center_map)

            h_para_vals, h_orth_vals = ScannIndex.compute_h_vals(X, self.T)
            
            X_frob = (X**2).sum()

            if ('descent' in kwargs and kwargs['descent']):#Refine with alternating min if descent is set to true.
                
                X_n = preprocessing.normalize(X)
                prev_cost = ((X - centers[center_map])**2).sum()
                
                for i in range(300):
                    old_center_map = center_map.copy()
                    self.update_centers(X, X_n, h_para_vals, h_orth_vals, centers, center_map)
                    self.update_centers_map(X, centers, center_map)
                    new_cost = ((X - centers[center_map])**2).sum()

                    #Break if mapping doesn't change or change in cost is smaller than relative threshold
                    if (all(center_map == old_center_map) or 
                        (np.abs(prev_cost - new_cost) <= ScannIndex.rel_tol*prev_cost)): 
                        break
                    
                    prev_cost = new_cost
                
                    if i == 299: print("Scann indexing did not converge")

            labels = scipy.sparse.csr_matrix((np.ones(X.shape[0]), (np.arange(X.shape[0]), center_map)),
                                             shape=(X.shape[0], self.k))
            
            if ('project' in kwargs and kwargs['project']): 
                self.compute_projections(X, labels, centers, h_para_vals, h_orth_vals)
                
            
        return (labels, centers)


    
    
    
    
class ScannProjIndex(ClusterIndex):
    atol_ip = 1e-3
    rel_tol = 1e-4
    def __init__(self, k, metric='angular', T=0.2): 
        ClusterIndex.__init__(self, metric)
        self.k = k
        self.T = T
        self.atol_ip = 1e-3
    
    def str_name(self, **kwargs): 
        if 'SQ' in kwargs: 
            return ("ScannProj_k=%d_SQ=%s" %(self.k, str(kwargs['SQ'])))
        else: 
            return ("ScannProj_k=%d" %(self.k))
    
    def compute_alpha(x, x_tilde, x_h_p, x_h_o):
        if (x_h_p == 0. or x_h_o == 0.): 
            alpha = 1.
        else:
            alpha = (2*x_h_p*x_tilde.dot(x))/(2*((x_h_p - x_h_o)*(x_tilde.dot(x)**2)/(x**2).sum() + x_h_o*((x_tilde**2).sum()))) 
        return alpha
    
    def compute_projections(self, X, center_map, centers, h_p, h_o): 
        alphas = np.zeros(len(center_map))
        for r in (range(center_map.shape[0])):
            cid = center_map[r]
            alphas[r] = ScannProjIndex.compute_alpha(X[r], centers[cid], h_p[r], h_o[r])
        return alphas
            
    
    def compute_new_center(cluster, cluster_n, h_para_vals, h_orth_vals):
        C_n = cluster_n
        C_n_hscaled = C_n*(np.sqrt(h_para_vals - h_orth_vals)[:, np.newaxis])
        A = (C_n_hscaled.T)@C_n_hscaled
        np.fill_diagonal(A, A.diagonal() + h_orth_vals.sum())
        return np.linalg.inv(A).dot((cluster*(h_para_vals[:, np.newaxis])).sum(axis=0))
        
    def update_centers_map(self, data, h_para_vals, h_orth_vals, centers, center_map):
        all_alphas = np.zeros((data.shape[0], centers.shape[0]))
        for c in range(centers.shape[0]):
            alphas = self.compute_projections(data, c*np.ones(data.shape[0], dtype=np.intc), centers, h_para_vals, h_orth_vals)
            all_alphas[:, c] = alphas
        alpha_dots = (data@centers.T)*all_alphas
        center_map[:] = np.argmin(-2*(alpha_dots) + (all_alphas**2)*(centers**2).sum(axis=1), axis=1)

    
    def update_centers(self, data, data_n, h_para_vals, h_orth_vals, centers, center_map):
        (k,d) = centers.shape
        for i in range(k):
            cluster_filter = center_map==i
            cluster = data[cluster_filter]
            if len(cluster) == 0:
                center = data[np.random.randint(data.shape[0])]
            else:
                norms = np.linalg.norm(cluster,ord=2,axis=1)
                if np.all(norms <= (self.T + ScannIndex.atol_ip)):
                    center = cluster[np.random.randint(cluster.shape[0])]
                else:
                    center = ScannProjIndex.compute_new_center(cluster, data_n[cluster_filter],
                                                       h_para_vals[cluster_filter], 
                                                       h_orth_vals[cluster_filter])
            centers[i,:] = center
                      
    
    def fit(self, X, **kwargs):
        
        h_para_vals, h_orth_vals = ScannIndex.compute_h_vals(X, self.T)
        
        #If k is larger than the |X| then initialize remainnig cetners to 0.
        if (self.k >= X.shape[0]): #Just return the labels and centers if k < |X|
            if len(X.shape) > 1: 
                centers = np.zeros((self.k, X.shape[1]))
            else: 
                centers = np.zeros(self.k)
            centers[:X.shape[0]] = X
            center_map = np.zeros(X.shape[0], dtype=np.intc)
            self.update_centers_map(X, h_para_vals, h_orth_vals, centers, center_map)
        else:         
            ixs = np.random.choice(X.shape[0], size=self.k, replace=False)
            centers = X[np.sort(ixs)] #CHoose random k centers from the data
            
            center_map = np.zeros(X.shape[0], dtype=np.intc)
            self.update_centers_map(X, h_para_vals, h_orth_vals, centers, center_map)
            
            X_frob = (X**2).sum()

            if ('descent' in kwargs and kwargs['descent']):#Refine with alternating min if descent is set to true.
                
                X_n = preprocessing.normalize(X)
                prev_cost = ((X - centers[center_map])**2).sum()
                
                for i in range(300):
                    old_center_map = center_map.copy()
                    self.update_centers(X, X_n, h_para_vals, h_orth_vals, centers, center_map)
                    self.update_centers_map(X, h_para_vals, h_orth_vals, centers, center_map)
                    new_cost = ((X - centers[center_map])**2).sum()

                    #Break if mapping doesn't change or change in cost is smaller than relative threshold
                    if (all(center_map == old_center_map) or 
                        (np.abs(prev_cost - new_cost) <= ScannProjIndex.rel_tol*prev_cost)): 
                        break
                    
                    prev_cost = new_cost
                
                    if i == 299: print("ScannProj indexing did not converge")
                        
        alphas = self.compute_projections(X, center_map, centers, h_para_vals, h_orth_vals)
        labels = scipy.sparse.csr_matrix((alphas, (np.arange(X.shape[0]), center_map)),
                                        shape=(X.shape[0], self.k))                
        return (labels, centers)



            
            

    
class ProjCIndex(ClusterIndex):
    def __init__(self, k, metric='angular'):
        ClusterIndex.__init__(self, metric)
        self.k = k
    
    def str_name(self, **kwargs): 
        if 'SQ' in kwargs: 
            return ("ProjC_k=%d_SQ=%s" %(self.k, str(kwargs['SQ'])))
        else: 
            return ("ProjC_k=%d" %self.k)
        
    
    def initialize_kmeans(X, k): 
        labels, centers = KMeansIndex.kmeans_mat(X, k)
        return(centers, labels)        


    def initialize_projective_sampling(X, k): 
        X_n = preprocessing.normalize(X)
        X_n_T = X_n.T 

        j = np.random.choice(len(X_n))
        centers = [X_n[j]] #initialize with far away centers
        centers_ixs = [j]
        
        cmap = np.zeros(X.shape[0])
        for i in range(k-1):
            #Compute residiual distances. Map every point to its closest center.
            res_dists = 2 - 2*(centers@X_n_T) # |a-b|^2 = a^2 - 2ab + b^2
            closest_map = np.argmin(res_dists, axis = 0)
            closest_dists = np.abs(res_dists[closest_map, np.arange(X_n_T.shape[1])])
            
            #Sample using residual distances as weights.
            j = np.random.choice(X_n_T.shape[1], 
                                 p=closest_dists/np.sum(closest_dists))
            centers_ixs.append(j)
            centers.append(X_n[j])

        return(X[np.array(centers_ixs)], np.argmin(2 - 2*(centers@X_n_T), axis=0))
            
            

    def update_centers_map(data, centers, center_map):
        # normalizing the centers
        centers /= np.linalg.norm(centers,axis=1)[:,np.newaxis]
        dots = data@centers.T
        center_map[:] = np.argmax(dots**2, axis=1)
    
    
    def update_centers(data, centers, center_map):
        (k,d) = centers.shape
        for i in range(k):
            cluster = data[center_map== i]
            if len(cluster) == 0:
                center = np.random.randn(d)
            else:
                center = PCAIndex.pca_mat(cluster, 1)[2][0]
            centers[i,:] = center
        centers /= np.linalg.norm(centers,axis=1)[:,np.newaxis]

    def fit(self, X, **kwargs):
        centers, center_map = kwargs['initializer'](X, self.k)

        if ('descent' in kwargs and kwargs['descent']):#Refine with alternating min if descent is set to true.
            for i in range(100):
                old_center_map = center_map.copy()
                ProjCIndex.update_centers(X, centers, center_map)
                ProjCIndex.update_centers_map(X, centers, center_map)
                if all(center_map == old_center_map):
                    break
            if i == 99: print("k-ray-means did not converge")
                
        centers /= np.linalg.norm(centers,axis=1)[:,np.newaxis]
        scalings = np.sum(centers[center_map]*X, axis=1)
        labels = scipy.sparse.csr_matrix((scalings, (np.arange(X.shape[0]), center_map)),
                                         shape=(X.shape[0], self.k))
        return (labels, centers)
        



class PCAIndex(ClusterIndex):
    def __init__(self, num_components, metric='angular'): 
        ClusterIndex.__init__(self, metric)
        self.num_components = num_components

    def str_name(self, **kwargs): 
        return ("PCA_k=%d" %num_components)
    
    '''Method to compute prinicipal components. Can be used in other indexing 
    classes.'''
    def pca_mat(X, num_components):
        return randomized_svd(X, n_components=num_components, random_state=0)    
    
    def fit(self, X, **kwargs):
        U, S, VT = PCAIndex.pca_mat(X, self.num_components)
        X_proj = S*U  #Multiply columns of U with S 
        return (scipy.sparse.csr_matrix(X_proj), VT)


    
    
    
'''Highest-level class of Logistic loss indexing class. Implements the query methods
but defers the fit method to subclasses depending on how assignments and clusters are to be trained.
Implements different functions for -- computing classification labels given queries'''
class LogisticLossIndex(ClusterIndex): 
    def __init__(self, int_dim, labels, metric='angular'):
        ClusterIndex.__init__(self, metric)
        self.int_dim = int_dim #number of clusters and dimension of assignments.
        self.labels = labels
    
    '''Coomputes X@Q and for every query point, a point in X is mapped to 1 
    if it is in top_k neighbors otherwise it is mapped to -1. Output is n x m.'''
    def compute_class_labels_topk(X, Q, top_k): 
        labels = np.zeros((Q.shape[0], X.shape[0]))
        topk_indices = np.argpartition(-Q@X.T, top_k, axis=-1)[:, :top_k]
        for i in range(labels.shape[0]): labels[i, topk_indices[i]] = 1
        return (2*labels - 1).T
    
    '''Coomputes X@Q and for every query point, a point in X is mapped to 1 
    if it is in top_k neighbors otherwise it is mapped to -1. Output is n x m.'''
    def compute_class_labels_threshold(X, Q, thresh): 
        return (2*((-Q@X.T <= -1.*thresh).astype(np.float32)) - 1).T
    
    '''Z is (self.centers.shape[0], train_set.shape[0])'''
    def find_center_fit(l, Z, thresh): 
        log_losses = np.zeros(Z.shape[0])
        ws = np.zeros(Z.shape[0])
        Z_t = -Z*l
        l_thresh = thresh*l
        def log_loss(w, i): return np.log(1 + np.exp(Z_t[i]*w + l_thresh)).sum()
        for i in range(Z.shape[0]): 
    #         res = minimize(lambda w: log_loss(w, i), 0., method='Nelder-Mead', tol=0.1e-3,
    #                                  options={'disp': True})
            res=LogisticRegression(C=10., penalty='l1', solver='saga').fit((Z[i]).reshape(-1, 1), l)
    #         log_losses[i] = log_loss(res.x,i)
            log_losses[i] = log_loss(res.coef_.ravel(), i)
    #         ws[i] = res.x
            ws[i] = res.coef_.ravel()
        i_star = np.argmin(log_losses)
        a = np.zeros(Z.shape[0])
        a[i_star] = ws[i_star]
        return a 
    



    
    
class LogisticLossFixedCentersIndex(LogisticLossIndex):
    def __init__(self, int_dim, thresh, labels, centers, metric='angular'): 
        LogisticLossIndex.__init__(self, int_dim=int_dim, 
                                   labels=labels, metric=metric)
        self.thresh = thresh
        self.centers = centers
    
    
    '''X is data matrix, clusters is k x d matrix of 
    cluster centers (i.e. B matrix), train_set is the set of 
    queries to be trained on, class_labels is the {-1, 1} labels of 
    the data on the queries train_set. Returns A matrix'''
    def fit(self, X, **kwargs): 
        train_set = kwargs['train_set']
        coeffs = scipy.sparse.lil_matrix((X.shape[0], self.centers.shape[0]))
        Z = train_set@self.centers.T
        for i in tqdm(range(X.shape[0])):
            #regress only if the point is classified as 1 for some point in train_set
            if np.sum(self.labels[i]) > -self.labels.shape[1]: 
                coeffs[i, :] = LogisticLossIndex.find_center_fit(self.labels[i], Z.T, self.thresh)
        return (coeffs.tocsr(), self.centers)
        

        
# class LogisticLossFixedCentersIndex(LogisticLossIndex):
#     def __init__(self, int_dim, train_set, labels, centers, inv_reg, pen, metric='angular'): 
#         LogisticLossIndex.__init__(self, int_dim=int_dim, train_set=train_set, 
#                                    labels=labels, metric=metric)
#         self.inv_reg = inv_reg
#         self.pen = pen
#         self.centers = centers
#         self.tot_score = 0.
        
#     '''X is data matrix, clusters is k x d matrix of 
#     cluster centers (i.e. B matrix), train_set is the set of 
#     queries to be trained on, class_labels is the {-1, 1} labels of 
#     the data on the queries train_set. Returns A matrix'''
#     def fit(self, X, train_set): 
#         coeffs = scipy.sparse.lil_matrix((X.shape[0], self.centers.shape[0]), dtype=np.float32)
#         Z = train_set@self.centers.T
#         tot_score = 0.
#         avg_sp = tqdm()
#         sp = 0.
#         for i in tqdm(range(X.shape[0])):
#             if np.sum(self.labels[i]) > -self.labels.shape[1]: #regress only if the point is classified as 1 for some point in train_set
#                 lr = LogisticRegression(C=self.inv_reg, penalty=self.pen, solver='saga').fit(Z, self.labels[i])
#                 coeffs[i, :] = lr.coef_.ravel()
# #                 tot_score += lr.score(Z, self.labels[i])

#             up = (sp*i+np.count_nonzero(coeffs[i])/coeffs.shape[1])/(i+1) - sp
#             sp += up            
#             avg_sp.update(up) #output avg_sparsity.
#         return (coeffs.tocsr(), self.centers)
        

        
# # t_X = np.array([[1, 2, 3, 4],[2, 4, 3, 4],[-1, -2, 3, 4],[-2, -4, 3, 4]])  
# t_X, _ = datasets.make_blobs(n_samples=4, n_features=10, centers=1, random_state=0)
# k_ix = KMeansPQIndex(2, 3)
# k_ix.build_index(t_X.astype(np.intc), np.zeros(10))
# print (t_X.astype(np.intc))
# print(k_ix.assignments)
# print(k_ix.centers)