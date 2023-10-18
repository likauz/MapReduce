# MapReduce
MapReduce - Multi-threaded Programming

MapReduce is used to parallelise tasks of a specific structure. 
Such tasks are defined by two functions, map and reduce, used as follows:

1) The input is given as a sequence of elements.
2) (Map phase) The map function is applied to each input element, producing a sequence of intermediary elements.
3) (Sort/Shuffle phases) The intermediary elements are sorted into new sequences (more on this later).
4) (Reduce phase) The reduce function is applied to each of the sorted sequences of intermediary elements, producing a sequence of output elements.
5) The output is a concatenation of all sequences of output elements.
