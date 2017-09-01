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
#define MEX_EDGE_TYPE_NUM 100

char node_file[MAX_STRING], link_file[MAX_STRING], output_file[MAX_STRING],meta_path_file[MAX_STRING],meta_path_head[1000],init_file[MAX_STRING];
int binary = 0, num_threads = 1, vector_size = 100, negative = 5, iters = 10, epoch, mode = 0, model = 0, depth = 2,meta_path_count = 0;
long long samples = 1, edge_count_actual;
real alpha = 0.025, starting_alpha, restart = 0.3;
real start_alpha;
int train_mode;
int rwr_seq,rwr_ppi;

int edge_type_num;

line_node node0, node1;
line_hin hin;
line_trainer_edge trainer_edge[10];
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
            printf("%cEpoch: %d/%d Alpha: %f Progress: %.3lf%% Error: %lf", 13, epoch + 1, iters, alpha, (real)edge_count_actual / (real)(samples + 1) * 100, error);
            fflush(stdout);
            alpha = starting_alpha * (1 - edge_count_actual / (real)(samples * iters+ 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
            error = 0;
        }
        for (int i=1;i<=edge_type_num;i++)
        {
            if (rwr_ppi==0)
            {
                error += trainer_edge[i].train_sample(mode, alpha,error_vec, error_p, error_q, func_rand_num, next_random);
            }
            else
            {
                error += trainer_edge[i].train_sample_depth(mode, alpha, depth,error_vec, error_p, error_q, func_rand_num, next_random);
                error += trainer_edge[i].train_sample_randwalk(mode, alpha, restart,error_vec, error_p, error_q, func_rand_num, next_random);
            }
        }
        for (int i=0;i<meta_path_count;i++)
        {
            error += trainer_path[i].train_path(mode, node_lst, alpha, error_vec, error_p, error_q, func_rand_num, next_random, model);
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
    printf("finished\n");
}

void TrainModel() {
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    starting_alpha = alpha;
    printf("what1?\n");
    fflush(stdout);
    node0.init(node_file, vector_size);
    node1.init(node_file, vector_size);
	fflush(stdout);
    hin.init(link_file, &node0, &node1);
    printf("waht %d\n",edge_type_num);
    for (int i=1;i<=edge_type_num;i++)
    {
        char edge_tp[MEX_EDGE_TYPE_NUM];
        printf("%d", i);
        sprintf(edge_tp, "%d", i);
        trainer_edge[i].init(edge_tp[0], &hin, negative);
    }
    printf("what2?");
    read_meta_path(meta_path_file);
    printf("what3?");
	fflush(stdout);
    /*
     trainer_edge8.init('8', &hin, negative);
     trainer_edge9.init('9', &hin, negative);
     
     trainer_path1.init("H1H", &hin, negative);
     trainer_path2.init("H2H", &hin, negative);
     trainer_path3.init("H3H", &hin, negative);
     trainer_path4.init("H4H", &hin, negative);
     trainer_path5.init("H5H", &hin, negative);
     trainer_path6.init("H6H", &hin, negative);
     */
	 /*
    clock_t start = clock();
    printf("Training:");
    if (train_mode == 0)
    {
        for (epoch = 0; epoch != iters; epoch++)
        {
            edge_count_actual = samples * epoch;
            mode = 3;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            write_to_file(epoch);
        }
    }
	
    if (train_mode == 1)
    {
        for (epoch = 0; epoch != iters; epoch++)
        {
            write_to_file(epoch);
            edge_count_actual = samples * epoch;
            mode = 0;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            
            mode = 1;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            mode = 2;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            
            
        }
    }
	*/
    if (train_mode == 2)
    {
        for (epoch = 0; epoch != iters; epoch++)
        {
            write_to_file(epoch);
						printf("what3?");
	fflush(stdout);
	
            edge_count_actual = samples * epoch;
            mode = 0;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
						printf("what3?");
	fflush(stdout);
            mode = 1;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
			printf("what3?");
	fflush(stdout);
        }
    }
	/*
    if (train_mode == 3)
    {
        for (epoch = 0; epoch != iters/50; epoch++)
        {
            for (int tmp_epoch = 0; tmp_epoch<25;tmp_epoch++)
            {
                edge_count_actual = samples * (epoch*25+tmp_epoch);
                mode = 1;
                for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
                for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            }
            
            for (int tmp_epoch = 0; tmp_epoch<25;tmp_epoch++)
            {
                edge_count_actual = samples * (epoch*25+tmp_epoch+25);
                mode = 2;
                for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
                for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            }
            write_to_file(epoch);
        }
    }
    
    if (train_mode == 4)
    {
        for (epoch = 0; epoch != iters/50; epoch++)
        {
            for (int tmp_epoch = 0; tmp_epoch<25;tmp_epoch++)
            {
                edge_count_actual = samples * (epoch*25+tmp_epoch);
                mode = 0;
                for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
                for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            }
            
            for (int tmp_epoch = 0; tmp_epoch<25;tmp_epoch++)
            {
                edge_count_actual = samples * (epoch*25+tmp_epoch+25);
                mode = 1;
                for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
                for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            }
            write_to_file(epoch);
        }
    }
    if (train_mode == 5)
    {
        for (epoch = 0; epoch != iters; epoch++)
        {
            edge_count_actual = samples * epoch;
            mode = 0;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            
            mode = 1;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            mode = 2;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            mode = 3;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            write_to_file(epoch);
        }
    }
    if (train_mode == 6)
    {
        for (epoch = 0; epoch != iters/50; epoch++)
        {
            for (int tmp_epoch = 0; tmp_epoch<50;tmp_epoch++)
            {
                edge_count_actual = samples * (epoch*50+tmp_epoch);
                mode = 0;
                for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
                for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
                mode = 1;
                for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
                for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            }
            
            mode = 2;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            
            write_to_file(epoch);
        }
    }
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
    
    node0.output(output_file, binary);
    printf("here main\n");
	*/
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
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
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
        printf("./hin2vec -node node.txt -link link.txt -path path.txt -output vec.emb -binary 1 -size 100 -negative 5 -samples 5 -iters 20 -threads 12\n\n");
        return 0;
    }
    output_file[0] = 0;
    
    if ((i = ArgPos((char *)"-node", argc, argv)) > 0) strcpy(node_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-meta_path", argc, argv)) > 0) strcpy(meta_path_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-link", argc, argv)) > 0) strcpy(link_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) samples = (long long)(atof(argv[i + 1])*1000000);
    if ((i = ArgPos((char *)"-iters", argc, argv)) > 0) iters = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-depth", argc, argv)) > 0) depth = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-restart", argc, argv)) > 0) restart = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-edge_type_num", argc, argv)) > 0) edge_type_num = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-rwr_ppi", argc, argv)) > 0) rwr_ppi = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-rwr_seq", argc, argv)) > 0) rwr_seq = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-train_mode", argc, argv)) > 0) train_mode = atoi(argv[i + 1]);	
    if ((i = ArgPos((char *)"-init_file", argc, argv)) > 0)  strcpy(init_file, argv[i + 1]);
    
    //start_alpha = alpha;
    //printf("edge_type_num=%d,mode=%d\n",edge_type_num,train_mode);
    /*
     string out_f(output_file);
     string tab = "_";
     out_f = out_f + tab+ out_dim + tab+out_dep+tab+sout_res;
     output_file = out_f.c_str();
     */
    
    
    TrainModel();
    printf("sfs\n");
    return 0;
}