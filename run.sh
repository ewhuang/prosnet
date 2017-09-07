#!/bin/sh

./embed -node /srv/data/simons_mouse/data/mouse_data/dca_networks_no_go/dca_genes_1.txt  -link /srv/data/simons_mouse/data/mouse_data/dca_networks_no_go/dca_edges_1.txt  -output ../result/dca_genes_1.vec -binary 0 -size 100 -negative 5 -samples 1 -iters 100 -threads 24 -model 2 -depth 10 -restart 0.8 -edge_type_num 3 -rwr_ppi 1 -rwr_seq 1 -train_mode 1
