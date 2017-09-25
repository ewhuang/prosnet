//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "linelib.h"
#include "ransampl.h"

#define MAX_PATH_LENGTH 100
#define MAX_EDGE_TYPE_NUM 100

char node_file[MAX_STRING], link_file[MAX_STRING], output_file[MAX_STRING], meta_path_file[MAX_STRING], meta_path_head[1000], init_file[MAX_STRING];
int binary = 0, num_threads = 1, vector_size = 100, negative = 5, iters = 10, epoch, mode = 0, model = 0, depth = 2, meta_path_count = 0;
long long samples = 1, edge_count_actual;
real alpha = 0.025, starting_alpha;
real start_alpha;
int edge_type_num;

line_node node0, node1;
line_hin hin;
line_trainer_edge trainer_edge[30];
line_trainer_path trainer_path[30];

#define SEED 12345
static unsigned int z1 = SEED, z2 = SEED, z3 = SEED, z4 = SEED;

double lfsr113_double(void)
{
   unsigned int b;
   b  = ((z1 << 6) ^ z1) >> 13;
   z1 = ((z1 & 4294967294UL) << 18) ^ b;
   b  = ((z2 << 2) ^ z2) >> 27; 
   z2 = ((z2 & 4294967288UL) << 2) ^ b;
   b  = ((z3 << 13) ^ z3) >> 21;
   z3 = ((z3 & 4294967280UL) << 7) ^ b;
   b  = ((z4 << 3) ^ z4) >> 12;
   z4 = ((z4 & 4294967168UL) << 13) ^ b;
   return (z1 ^ z2 ^ z3 ^ z4) * 2.3283064365386963e-10;
}

double func_rand_num()
{
    return lfsr113_double();
}

void *training_thread(void *id)
{
    long long edge_count = 0, last_edge_count = 0;
    unsigned long long next_random = (long long)id;
    real *error_vec = (real *)calloc(vector_size, sizeof(real));
    real *error_p = (real *)calloc(vector_size, sizeof(real));
    real *error_q = (real *)calloc(vector_size, sizeof(real));
    int *node_lst = (int *)malloc(MAX_PATH_LENGTH * sizeof(int));
    double error = 0;
    
    while (1)
    {
        //judge for exit
        if (edge_count > samples / num_threads + 2) break;
        
        if (edge_count - last_edge_count > 1000)
        {
            error = -error / (edge_count - last_edge_count);
            // if error is smaller than a threshold, then break.
            // if error does not decrease, then decease the learning rate alpha.
            edge_count_actual += edge_count - last_edge_count;
            last_edge_count = edge_count;
            printf("%cEpoch: %d/%d Alpha: %f Progress: %.3lf%% Error: %lf", 13,
                epoch + 1, iters, alpha, (real)edge_count_actual /
                (real)(samples + 1) * 100, log(error));
            fflush(stdout);
            alpha = starting_alpha * (1 - edge_count_actual / (real)(samples * iters+ 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
            error = 0;
        }
        for (int i=1;i<=20;i++)
        {
            for (int j=1;j<=edge_type_num;j++) {
                error += trainer_edge[j].train_sample(mode, alpha,error_vec, error_p,
                    error_q, func_rand_num, next_random);
            }
        }
		for (int j=1;j<=20;j++)
        for (int i=0;i<meta_path_count;i++)
        {
            error += trainer_path[i].train_path(mode, node_lst, alpha, error_vec,
                error_p, error_q, func_rand_num, next_random, model);
        }
        
        edge_count += edge_type_num;
    }
    free(node_lst);
    free(error_vec);
    free(error_p);
    free(error_q);
    pthread_exit(NULL);
}

void read_meta_path(char *file_name)
{
    FILE *fi = fopen(file_name, "rb");
    if (fi == NULL)
    {
        printf("ERROR: meta path file not found!\n");
        printf("%s\n", file_name);
        exit(1);
    }
    meta_path_count = 0;
    char path[MAX_STRING];
    while (1)
    {
        if (fscanf(fi, "%s", path) != 1) break;
        trainer_path[meta_path_count].init(path, &hin, negative);
        sprintf(meta_path_head, "%s_%s", meta_path_head,path);
        meta_path_count++;
    }
    fclose(fi);
    
}

void write_to_file(int epoch)
{
    char new_output_file[MAX_STRING];
    sprintf(new_output_file, "%s_%d", output_file,epoch);
    printf("%s\n",new_output_file);
    node0.output(new_output_file, binary);
}

char nth_letter(int n)
{
    assert(n >= 1 && n <= 26);
    return "abcdefghijklmnopqrstuvwxyz"[n-1];
}

void TrainModel() {
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    starting_alpha = alpha;

    node0.init(node_file, vector_size);
    node1.init(node_file, vector_size);

    hin.init(link_file, &node0, &node1);

    for (int i=1;i<=edge_type_num;i++)
    {
        char edge_tp[MAX_EDGE_TYPE_NUM];
        sprintf(edge_tp, "%d", i);
        trainer_edge[i].init(nth_letter(i), &hin, negative);
    }

    read_meta_path(meta_path_file);

    for (epoch = 0; epoch != iters; epoch++)
    {
        if (epoch == iters - 1) {
            write_to_file(epoch);
        }

        edge_count_actual = samples * epoch;
        mode = 0;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

        mode = 1;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    }
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("HIN2VEC\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-node <file>\n");
        printf("\t\tA dictionary of all nodes\n");
        printf("\t-link <file>\n");
        printf("\t\tAll links between nodes. Links are directed.\n");
        printf("\t-path <int>\n");
        printf("\t\tAll meta-paths. One path per line.\n");
        printf("\t-output <int>\n");
        printf("\t\tThe output file.\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 5 - 10 (0 = not used)\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int>Million\n");
        printf("\t-iters <int>\n");
        printf("\t\tSet the number of interations.\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\nExamples:\n");
        printf("./hin2vec -node node.txt -link link.txt -path path.txt -output vec.emb -size 100 -negative 5 -samples 5 -iters 20 -threads 12\n\n");
        return 0;
    }
    output_file[0] = 0;
    
    if ((i = ArgPos((char *)"-node", argc, argv)) > 0) strcpy(node_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-meta_path", argc, argv)) > 0) strcpy(meta_path_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-link", argc, argv)) > 0) strcpy(link_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) samples = (long long)(atof(argv[i + 1])*1000000);
    if ((i = ArgPos((char *)"-iters", argc, argv)) > 0) iters = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-depth", argc, argv)) > 0) depth = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-edge_type_num", argc, argv)) > 0) edge_type_num = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-init_file", argc, argv)) > 0)  strcpy(init_file, argv[i + 1]);
    
    TrainModel();
    return 0;
}