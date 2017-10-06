# ProSNet
ProSNet heterogenous network embedding algorithm.

### Running ProSNet.

ProSNet comes with a Python wrapper file for ease of use.

```bash
python prosnet.py [-h] -d NUM_DIM -n NODE_FNAME -e EDGE_FNAME -m META_FNAME -o
                  OUT_FNAME -i NUM_ITERATIONS
```

It simply requires five input arguments.

1. -d is the number of desired embedding vector dimensions.
2. -n is the name of the node file. It should be a two column, tab-separated file, where the first column is the name of the node, and the second column is the name of the node type.
3. -e is the name of the edge file. It should be a four column, tab-separated file, where the first two columns are the names of the nodes in each edge, the third column is the edge weight, and the fourth column is the type of edge. Edges should be added in forwards and backwards to create an undirected edge.
4. -o is the name of the output file format.
5. -i is the number of iterations for which to run ProSNet.
