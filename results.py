from index_class import data_io 

class Results: 
    def __init__(self, cluster_indexes_types, results, **kwargs):
        self.index_types = cluster_indexes_types
        self.results = results
        self.kwargs = kwargs
    def results_filename(label): 
        return "results/Results_%s"%label
    def save_results(self, label): 
        data_io.save_pickle(Results.results_filename(label), self)
    def load_results(label): 
        return data_io.load_pickle(Results.results_filename(label))
