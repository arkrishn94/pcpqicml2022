import pickle 


def save_pickle(fn, data): 
    outfile = open(fn,'wb')
    pickle.dump(data, outfile)
    outfile.close()
    
def load_pickle(fn):
    infile = open(fn,'rb')
    data = pickle.load(infile)
    infile.close()
    return data
