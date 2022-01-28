# Projective Clustering Product Quantization 

### Instructions

To generate figures use iPython notebook `PCPQ-Recall.ipynb`. 

Indexing using 4-bit encoding, i.e. $k = 16$,  takes between 1-10 minutes for k-means and up to 1 hour for `Q-PCPQ` and `Q-APCPQ`. 

Indexing using 8-bit encoding can take up to 10-12 hours for `Q-PCPQ` and `Q-APCPQ`. 



### Repository Contents 

* `index_class/`: Core functions to create each index 

* `results`: Directory to store indexes

* `results.py`: Functions for loading and saving recall results

* `test.py`: Functions for computing recall of indexes. 

  





