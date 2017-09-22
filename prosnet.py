import argparse
import os
import subprocess

def run_prosnet(num_edge_types):
    command = ('./embed -node %s -link %s -meta_path %s -output %s -size %d '
        '-negative 5 -samples 1 -iters %d -threads 24 -depth 10 '
        '-edge_type_num %d' % (args.node_fname,
            args.edge_fname, args.meta_fname, args.out_fname, args.num_dim,
            args.num_iterations, num_edge_types))
    print command
    subprocess.call(command, shell=True)

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--num_dim', help='Number of embedding dimensions.',
        required=True, type=int)
    parser.add_argument('-n', '--node_fname', help='File name for nodes.',
        required=True, type=str)
    parser.add_argument('-e', '--edge_fname', help='File name for edges.',
        required=True, type=str)
    parser.add_argument('-m', '--meta_fname', help='File name for meta-paths.',
        required=True, type=str)
    parser.add_argument('-o', '--out_fname', help='File name for output vectors.',
        required=True, type=str)
    parser.add_argument('-i', '--num_iterations', help='Number of iterations.',
        required=True, type=int)
    
    args = parser.parse_args()

    folder = '/'.join(args.out_fname.split('/')[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder)

def check_edge_file():
    '''
    Ensures that the all nodes that appear in the edge file also appear in the
    node file.
    '''
    node_set, edge_type_set = set([]), set([])

    # Read the node file.
    f = open(args.node_fname, 'r')
    for line in f:
        node, node_type = line.strip().split('\t')
        node_set.add(node)        
    f.close()

    # Read the edge file.
    f = open(args.edge_fname, 'r')
    for line in f:
        node_a, node_b, edge_weight, edge_type = line.strip().split('\t')
        # Add to the set of edge types.
        edge_type_set.add(edge_type)
        # Check that each node is already in the node set.
        assert node_a in node_set and node_b in node_set
    f.close()

    return len(edge_type_set)

def main():
    parse_args()

    # Sanity checks.
    num_edge_types = check_edge_file()

    run_prosnet(num_edge_types)

if __name__ == '__main__':
    main()