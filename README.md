# Biggraph

To train an embedding, modify the *config.py* file.

Data should be located in the folder "/data/graphs/".

Input graphs should be stored in tab separated values. To generate such a graph, you can use for example the following command 

    java -cp "webgraph/*" -server it.unimi.dsi.webgraph.ArcListASCIIGraph graphs/cnr-2000/cnr-2000 graphs/cnr-2000/cnr-2000.tab

