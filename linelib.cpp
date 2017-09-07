#include "linelib.h"

std::vector<line_trans *> line_trainer_edge::vec_trans = std::vector<line_trans *>();
std::map<std::string, int> line_trainer_edge::map_trans = std::map<std::string, int>();
int line_trainer_edge::cnt_trans = 0;

std::vector<line_trans *> line_trainer_path::vec_trans = std::vector<line_trans *>();
std::map<std::string, int> line_trainer_path::map_trans = std::map<std::string, int>();
int line_trainer_path::cnt_trans = 0;

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

line_node::line_node() : vec(NULL, 0, 0)
{
    node = NULL;
    node_size = 0;
    node_max_size = 1000;
    vector_size = 0;
    node_file[0] = 0;
    node_hash = NULL;
    _vec = NULL;
}

line_node::~line_node()
{
    if (node != NULL) {free(node); node = NULL;}
    node_size = 0;
    node_max_size = 0;
    vector_size = 0;
    node_file[0] = 0;
    if (node_hash != NULL) {free(node_hash); node_hash = NULL;}
    if (_vec != NULL) {free(_vec); _vec = NULL;}
    new (&vec) Eigen::Map<BLPMatrix>(NULL, 0, 0);
}

int line_node::get_hash(char *word)
{
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % hash_table_size;
    return hash;
}

int line_node::search(char *word)
{
    unsigned int hash = get_hash(word);
    while (1) {
        if (node_hash[hash] == -1) return -1;
        if (!strcmp(word, node[node_hash[hash]].word)) return node_hash[hash];
        hash = (hash + 1) % hash_table_size;
    }
    return -1;
}

int line_node::add_node(char *word, char type)
{
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    node[node_size].word = (char *)calloc(length, sizeof(char));
    strcpy(node[node_size].word, word);
    node[node_size].type = type;
    node_size++;
    // Reallocate memory if needed
    if (node_size + 2 >= node_max_size) {
        node_max_size += 1000;
        node = (struct struct_node *)realloc(node, node_max_size * sizeof(struct struct_node));
    }
    hash = get_hash(word);
    while (node_hash[hash] != -1) hash = (hash + 1) % hash_table_size;
    node_hash[hash] = node_size - 1;
    return node_size - 1;
}

void line_node::init(char *file_name, int vector_dim)
{
    strcpy(node_file, file_name);
    vector_size = vector_dim;
    
    node = (struct struct_node *)calloc(node_max_size, sizeof(struct struct_node));
    node_hash = (int *)calloc(hash_table_size, sizeof(int));
    for (int k = 0; k != hash_table_size; k++) node_hash[k] = -1;
    
    FILE *fi = fopen(node_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: node file not found!\n");
        printf("%s\n", node_file);
        exit(1);
    }
    
    char word[MAX_STRING], type;
    node_size = 0;
    while (1)
    {
        if (fscanf(fi, "%s %c", word, &type) != 2) break;
        //int cn;
        //if (fscanf(fi, "%s %d %c", word, &cn, &type) != 3) break;
        //if (fscanf(fi, "%s", word) != 1) break;
        //type = 'w';
        add_node(word, type);
    }
    fclose(fi);
    
    long long a, b;
    a = posix_memalign((void **)&_vec, 128, (long long)node_size * vector_size * sizeof(real));
    if (_vec == NULL) { printf("Memory allocation failed\n"); exit(1); }
    for (b = 0; b < vector_size; b++) for (a = 0; a < node_size; a++)
        _vec[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    new (&vec) Eigen::Map<BLPMatrix>(_vec, node_size, vector_size);
    
    printf("Reading nodes from file: %s, DONE!\n", node_file);
    printf("Node size: %d\n", node_size);
    printf("Node dims: %d\n", vector_size);
}

void line_node::output(char *file_name, int binary)
{
    FILE *fo = fopen(file_name, "wb");
    fprintf(fo, "%d %d\n", node_size, vector_size);
    for (int a = 0; a != node_size; a++)
    {
        fprintf(fo, "%s ", node[a].word);
        if (binary) for (int b = 0; b != vector_size; b++) fwrite(&_vec[a * vector_size + b], sizeof(real), 1, fo);
        else for (int b = 0; b != vector_size; b++) fprintf(fo, "%lf ", _vec[a * vector_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

line_hin::line_hin()
{
    hin_file[0] = 0;
    node_u = NULL;
    node_v = NULL;
    hin = NULL;
    hin_size = 0;
}

line_hin::~line_hin()
{
    hin_file[0] = 0;
    node_u = NULL;
    node_v = NULL;
    if (hin != NULL) {delete [] hin; hin = NULL;}
    hin_size = 0;
}

void line_hin::init(char *file_name, line_node *p_u, line_node *p_v)
{
    strcpy(hin_file, file_name);
    
    node_u = p_u;
    node_v = p_v;
    
    int node_size = node_u->node_size;
    hin = new std::vector<hin_nb>[node_size];
    
    FILE *fi = fopen(hin_file, "rb");
    char word1[MAX_STRING], word2[MAX_STRING], tp;
    int u, v;
    double w;
    hin_nb curnb;
    while (fscanf(fi, "%s %s %lf %c", word1, word2, &w, &tp) == 4)
    {
        if (hin_size % 10000 == 0)
        {
            printf("%lldK%c", hin_size / 1000, 13);
            fflush(stdout);
        }
        
        u = node_u->search(word1);
        v = node_v->search(word2);
        
        if (u != -1 && v != -1)
        {
            curnb.nb_id = v;
            curnb.eg_tp = tp;
            curnb.eg_wei = w;
            hin[u].push_back(curnb);
            hin_size++;
        }
    }
    fclose(fi);
    
    printf("Reading edges from file: %s, DONE!\n", hin_file);
    printf("%lld\n",hin_size+1);
    printf("Edge size: %lld\n", hin_size);
    printf("Edge size: %lld\n", hin_size);
    printf("Edge size: %lld\n", hin_size);
    printf("%lld\n",hin_size+1);
    printf("Reading edges from file: %s, DONE!\n", hin_file);
    printf("123");
    printf("Reading edges from file: %s, DONE!\n", hin_file);
    printf("123");
    printf("123");
}

line_trans::line_trans() : P(NULL, 0), Q(NULL, 0)
{
    type = "";
    vector_size = 0;
    _P = NULL;
    _Q = NULL;
    bias = 0;
}

line_trans::~line_trans()
{
    type = "";
    vector_size = 0;
    if (_P != NULL) {free(_P); _P = NULL;}
    if (_Q != NULL) {free(_Q); _Q = NULL;}
    new (&P) Eigen::Map<BLPVector>(NULL, 0);
    new (&Q) Eigen::Map<BLPVector>(NULL, 0);
    bias = 0;
}

void line_trans::init(std::string map_type, int vector_dim)
{
    type = map_type;
    vector_size = vector_dim;
    _P = (real *)malloc(vector_size * sizeof(real));
    _Q = (real *)malloc(vector_size * sizeof(real));
    
    for (int a = 0; a < vector_size; a++) _P[a] = 0;
    new (&P) Eigen::Map<BLPVector>(_P, vector_size);
    for (int a = 0; a < vector_size; a++) _Q[a] = 0;
    new (&Q) Eigen::Map<BLPVector>(_Q, vector_size);
}


line_trainer_edge::line_trainer_edge()
{
    edge_tp = 0;
    phin = NULL;
    expTable = NULL;
    u_nb_cnt = NULL;
    u_nb_id = NULL;
    u_nb_wei = NULL;
    u_wei = NULL;
    v_wei = NULL;
    smp_u = NULL;
    smp_u_nb = NULL;
    expTable = NULL;
    neg_samples = 0;
    neg_table = NULL;
}

line_trainer_edge::~line_trainer_edge()
{
    edge_tp = 0;
    phin = NULL;
    if (expTable != NULL) {free(expTable); expTable = NULL;}
    if (u_nb_cnt != NULL) {free(u_nb_cnt); u_nb_cnt = NULL;}
    if (u_nb_id != NULL) {free(u_nb_id); u_nb_id = NULL;}
    if (u_nb_wei != NULL) {free(u_nb_wei); u_nb_wei = NULL;}
    if (u_wei != NULL) {free(u_wei); u_wei = NULL;}
    if (v_wei != NULL) {free(v_wei); v_wei = NULL;}
    if (smp_u != NULL)
    {
        ransampl_free(smp_u);
        smp_u = NULL;
    }
    if (smp_u_nb != NULL)
    {
        free(smp_u_nb);
        smp_u_nb = NULL;
    }
    neg_samples = 0;
    if (neg_table != NULL) {free(neg_table); neg_table = NULL;}
}

void line_trainer_edge::init(char edge_type, line_hin *p_hin, int negative)
{
    edge_tp = edge_type;
    phin = p_hin;
    neg_samples = negative;
    line_node *node_u = phin->node_u, *node_v = phin->node_v;
    if (node_u->vector_size != node_v->vector_size)
    {
        printf("ERROR: vector dimsions are not same!\n");
        exit(1);
    }
    
    // compute the degree of vertices
    u_nb_cnt = (int *)calloc(node_u->node_size, sizeof(int));
    u_wei = (double *)calloc(node_u->node_size, sizeof(double));
    v_wei = (double *)calloc(node_v->node_size, sizeof(double));
    for (int u = 0; u != node_u->node_size; u++)
    {
        for (int k = 0; k != (int)(phin->hin[u].size()); k++)
        {
            int v = phin->hin[u][k].nb_id;
            char cur_edge_type = phin->hin[u][k].eg_tp;
            double wei = phin->hin[u][k].eg_wei;
            
            if (cur_edge_type != edge_tp) continue;
            
            u_nb_cnt[u]++;
            u_wei[u] += wei;
            v_wei[v] += wei;
        }
    }
    
    // allocate spaces for edges
    u_nb_id = (int **)malloc(node_u->node_size * sizeof(int *));
    u_nb_wei = (double **)malloc(node_u->node_size * sizeof(double *));
    for (int k = 0; k != node_u->node_size; k++)
    {
        u_nb_id[k] = (int *)malloc(u_nb_cnt[k] * sizeof(int));
        u_nb_wei[k] = (double *)malloc(u_nb_cnt[k] * sizeof(double));
    }
    
    // read neighbors
    int *pst = (int *)calloc(node_u->node_size, sizeof(int));
    for (int u = 0; u != node_u->node_size; u++)
    {
        for (int k = 0; k != (int)(phin->hin[u].size()); k++)
        {
            int v = phin->hin[u][k].nb_id;
            char cur_edge_type = phin->hin[u][k].eg_tp;
            double wei = phin->hin[u][k].eg_wei;
            
            if (cur_edge_type != edge_tp) continue;
            
            u_nb_id[u][pst[u]] = v;
            u_nb_wei[u][pst[u]] = wei;
            pst[u]++;
        }
    }
    free(pst);
    
    // init sampler for edges
    smp_u = ransampl_alloc(node_u->node_size);
    ransampl_set(smp_u, u_wei);
    smp_u_nb = (ransampl_ws **)malloc(node_u->node_size * sizeof(ransampl_ws *));
    for (int k = 0; k != node_u->node_size; k++)
    {
        if (u_nb_cnt[k] == 0) continue;
        smp_u_nb[k] = ransampl_alloc(u_nb_cnt[k]);
        ransampl_set(smp_u_nb[k], u_nb_wei[k]);
    }
    
    // Init negative sampling table
    neg_table = (int *)malloc(neg_table_size * sizeof(int));
    
    int a, i;
    double total_pow = 0, d1;
    double power = 0.75;
    for (a = 0; a < node_v->node_size; a++) total_pow += pow(v_wei[a], power);
    a = 0; i = 0;
    d1 = pow(v_wei[i], power) / (double)total_pow;
    while (a < neg_table_size) {
        if ((a + 1) / (double)neg_table_size > d1) {
            i++;
            if (i >= node_v->node_size) {i = node_v->node_size - 1; d1 = 2;}
            d1 += pow(v_wei[i], power) / (double)total_pow;
        }
        else
            neg_table[a++] = i;
    }
    
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    
    // add trans
    std::string tp = std::string();
    tp += edge_tp;
    if (line_trainer_edge::map_trans[tp] == 0)
    {
        line_trans *ptrans = new line_trans;
        ptrans->init(tp, phin->node_u->vector_size);
        line_trainer_edge::vec_trans.push_back(ptrans);
        line_trainer_edge::map_trans[tp] = line_trainer_edge::cnt_trans + 1;
        line_trainer_edge::cnt_trans++;
    }
}

void line_trainer_edge::print_trans()
{
    std::map<std::string, int>::iterator iter;
    for (iter = line_trainer_edge::map_trans.begin(); iter != line_trainer_edge::map_trans.end(); iter++)
    {
        std::string tp = iter->first;
        int id = iter->second;
        printf("%s %d %s\n", tp.c_str(), id, line_trainer_edge::vec_trans[id - 1]->type.c_str());
    }
}

void line_trainer_edge::copy_neg_table(line_trainer_edge *p_trainer_line)
{
    if (phin->node_v->node_size != p_trainer_line->phin->node_v->node_size)
    {
        printf("ERROR: node sizes are not same!\n");
        exit(1);
    }
    
    int node_size = phin->node_v->node_size;
    
    for (int k = 0; k != node_size; k++) v_wei[k] = p_trainer_line->v_wei[k];
    for (int k = 0; k != neg_table_size; k++) neg_table[k] = p_trainer_line->neg_table[k];
}

double line_trainer_edge::train_sample(int mode, real alpha, real *_error_vec, real *_error_p, real *_error_q, double (*func_rand_num)(), unsigned long long &rand_index)
{
    double logsum = 0;
    
    int target, label, u, v, index, vector_size, trans_id;
    real f, g;
    std::string edge_type = std::string();
    line_node *node_u = phin->node_u, *node_v = phin->node_v;
    
    u = (int)(ransampl_draw(smp_u, func_rand_num(), func_rand_num()));
    if (u_nb_cnt[u] == 0) return 0;
    index = (int)(ransampl_draw(smp_u_nb[u], func_rand_num(), func_rand_num()));
    v = u_nb_id[u][index];
    
    vector_size = node_u->vector_size;
    Eigen::Map<BLPVector> error_vec(_error_vec, vector_size);
    Eigen::Map<BLPVector> error_p(_error_p, vector_size);
    Eigen::Map<BLPVector> error_q(_error_q, vector_size);
    error_vec.setZero();
    error_p.setZero();
    error_q.setZero();
    
    edge_type += edge_tp;
    trans_id = map_trans[edge_type];
    if (trans_id == 0)
    {
        printf("ERROR: trans id error!\n");
        exit(1);
    }
    trans_id--;
    
    for (int d = 0; d < neg_samples + 1; d++)
    {
        if (d == 0)
        {
            target = v;
            label = 1;
        }
        else
        {
            rand_index = rand_index * (unsigned long long)25214903917 + 11;
            target = neg_table[(rand_index >> 16) % neg_table_size];
            if (target == v) continue;
            label = 0;
        }
        f = 0;
        f += node_u->vec.row(u) * node_v->vec.row(target).transpose();
        if (mode == 1 || mode == 2 || mode == 3)
        {
            f += (node_u->vec.row(u)) * (vec_trans[trans_id]->P.transpose());
            f += (node_v->vec.row(target)) * (vec_trans[trans_id]->Q.transpose());
            f += vec_trans[trans_id]->bias;
        }
        
        if (label == 1) logsum += log(sigmoid(f));
        else logsum += log(1 - sigmoid(f));
        
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        if (mode == 2 || mode == 3)
        {
            error_p += g * (node_u->vec.row(u));
            error_q += g * (node_v->vec.row(target));
        }
        if (mode == 1 || mode == 3)
        {
            error_vec += g * ((node_v->vec.row(target)));
            error_vec += g * (vec_trans[trans_id]->P);
            
            node_v->vec.row(target) += g * ((node_u->vec.row(u)));
            node_v->vec.row(target) += g * (vec_trans[trans_id]->Q);
        }
        if (mode == 0)
        {
            error_vec += g * node_v->vec.row(target);
            node_v->vec.row(target) += g * node_u->vec.row(u);
        }
    }
    if (mode == 0 || mode == 1 || mode == 3) node_u->vec.row(u) += error_vec;
    if (mode == 2 || mode == 3)
    {
        vec_trans[trans_id]->P += error_p;
        vec_trans[trans_id]->Q += error_q;
    }
    new (&error_vec) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    new (&error_p) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    new (&error_q) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    
    return logsum / (1 + neg_samples);
}

double line_trainer_edge::train_sample_depth(int mode, real alpha, int depth, real *_error_vec, real *_error_p, real *_error_q, double (*func_rand_num)(), unsigned long long &rand_index)
{
    double logsum = 0;
    
    int target, label, u, v, node, index, vector_size, trans_id;
    real f, g;
    std::string edge_type = std::string();
    line_node *node_u = phin->node_u, *node_v = phin->node_v;
    std::vector<int> node_lst;
    int node_cnt = 0;
    
    vector_size = node_u->vector_size;
    Eigen::Map<BLPVector> error_vec(_error_vec, vector_size);
    Eigen::Map<BLPVector> error_p(_error_p, vector_size);
    Eigen::Map<BLPVector> error_q(_error_q, vector_size);
    error_vec.setZero();
    error_p.setZero();
    error_q.setZero();
    
    edge_type += edge_tp;
    trans_id = map_trans[edge_type];
    if (trans_id == 0)
    {
        printf("ERROR: trans id error!\n");
        exit(1);
    }
    trans_id--;
    
    node_lst.clear();
    
    node = (int)(ransampl_draw(smp_u, func_rand_num(), func_rand_num()));
    node_lst.push_back(node);
    node_cnt = 0;
    while(1)
    {
        if (u_nb_cnt[node_lst[node_cnt]] == 0) break;
        index = (int)(ransampl_draw(smp_u_nb[node_lst[node_cnt]], func_rand_num(), func_rand_num()));
        node = u_nb_id[node_lst[node_cnt]][index];
        node_lst.push_back(node);
        node_cnt++;
        if (node_cnt == depth) break;
    }
    
    for (int k = 1; k <= node_cnt; k++)
    {
        u = node_lst[0];
        v = node_lst[k];
        
        error_vec.setZero();
        error_p.setZero();
        error_q.setZero();
        
        for (int d = 0; d < neg_samples + 1; d++)
        {
            if (d == 0)
            {
                target = v;
                label = 1;
            }
            else
            {
                rand_index = rand_index * (unsigned long long)25214903917 + 11;
                target = neg_table[(rand_index >> 16) % neg_table_size];
                if (target == v) continue;
                label = 0;
            }
            f = 0;
            f += node_u->vec.row(u) * node_v->vec.row(target).transpose();
            if (mode == 1 || mode == 2 || mode == 3)
            {
                f += (node_u->vec.row(u)) * (vec_trans[trans_id]->P.transpose());
                f += (node_v->vec.row(target)) * (vec_trans[trans_id]->Q.transpose());
                f += vec_trans[trans_id]->bias;
            }
            
            if (label == 1) logsum += log(sigmoid(f));
            else logsum += log(1 - sigmoid(f));
            
            if (f > MAX_EXP) g = (label - 1) * alpha;
            else if (f < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            if (mode == 2 || mode == 3)
            {
                error_p += g * (node_u->vec.row(u));
                error_q += g * (node_v->vec.row(target));
            }
            if (mode == 1 || mode == 3)
            {
                error_vec += g * ((node_v->vec.row(target)));
                error_vec += g * (vec_trans[trans_id]->P);
                
                node_v->vec.row(target) += g * ((node_u->vec.row(u)));
                node_v->vec.row(target) += g * (vec_trans[trans_id]->Q);
            }
            if (mode == 0)
            {
                error_vec += g * node_v->vec.row(target);
                node_v->vec.row(target) += g * node_u->vec.row(u);
            }
        }
        if (mode == 0 || mode == 1 || mode == 3) node_u->vec.row(u) += error_vec;
        if (mode == 2 || mode == 3)
        {
            vec_trans[trans_id]->P += error_p;
            vec_trans[trans_id]->Q += error_q;
        }
    }
    new (&error_vec) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    new (&error_p) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    new (&error_q) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    
    return logsum / (1 + neg_samples) / node_cnt;
}

double line_trainer_edge::train_sample_randwalk(int mode, real alpha, real restart, real *_error_vec, real *_error_p, real *_error_q, double (*func_rand_num)(), unsigned long long &rand_index)
{
    double logsum = 0;
    
    int target, label, u, v, node, index, vector_size, trans_id;
    real f, g;
    std::string edge_type = std::string();
    line_node *node_u = phin->node_u, *node_v = phin->node_v;
    std::vector<int> node_lst;
    int node_cnt = 0;
    
    vector_size = node_u->vector_size;
    Eigen::Map<BLPVector> error_vec(_error_vec, vector_size);
    Eigen::Map<BLPVector> error_p(_error_p, vector_size);
    Eigen::Map<BLPVector> error_q(_error_q, vector_size);
    error_vec.setZero();
    error_p.setZero();
    error_q.setZero();
    
    edge_type += edge_tp;
    trans_id = map_trans[edge_type];
    if (trans_id == 0)
    {
        printf("ERROR: trans id error!\n");
        exit(1);
    }
    trans_id--;
    
    node_lst.clear();
    
    node = (int)(ransampl_draw(smp_u, func_rand_num(), func_rand_num()));
    node_lst.push_back(node);
    node_cnt = 0;
    while(1)
    {
        if (func_rand_num() < restart) break;
        if (u_nb_cnt[node_lst[node_cnt]] == 0) break;
        index = (int)(ransampl_draw(smp_u_nb[node_lst[node_cnt]], func_rand_num(), func_rand_num()));
        node = u_nb_id[node_lst[node_cnt]][index];
        node_lst.push_back(node);
        node_cnt++;
    }
    
    for (int k = 1; k <= node_cnt; k++)
    {
        u = node_lst[0];
        v = node_lst[k];
        
        error_vec.setZero();
        error_p.setZero();
        error_q.setZero();
        
        for (int d = 0; d < neg_samples + 1; d++)
        {
            if (d == 0)
            {
                target = v;
                label = 1;
            }
            else
            {
                rand_index = rand_index * (unsigned long long)25214903917 + 11;
                target = neg_table[(rand_index >> 16) % neg_table_size];
                if (target == v) continue;
                label = 0;
            }
            f = 0;
            f += node_u->vec.row(u) * node_v->vec.row(target).transpose();
            if (mode == 1 || mode == 2 || mode == 3)
            {
                f += (node_u->vec.row(u)) * (vec_trans[trans_id]->P.transpose());
                f += (node_v->vec.row(target)) * (vec_trans[trans_id]->Q.transpose());
                f += vec_trans[trans_id]->bias;
            }
            
            if (label == 1) logsum += log(sigmoid(f));
            else logsum += log(1 - sigmoid(f));
            
            if (f > MAX_EXP) g = (label - 1) * alpha;
            else if (f < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            if (mode == 2 || mode == 3)
            {
                error_p += g * (node_u->vec.row(u));
                error_q += g * (node_v->vec.row(target));
            }
            if (mode == 1 || mode == 3)
            {
                error_vec += g * ((node_v->vec.row(target)));
                error_vec += g * (vec_trans[trans_id]->P);
                
                node_v->vec.row(target) += g * ((node_u->vec.row(u)));
                node_v->vec.row(target) += g * (vec_trans[trans_id]->Q);
            }
            if (mode == 0)
            {
                error_vec += g * node_v->vec.row(target);
                node_v->vec.row(target) += g * node_u->vec.row(u);
            }
        }
        if (mode == 0 || mode == 1 || mode == 3) node_u->vec.row(u) += error_vec;
        if (mode == 2 || mode == 3)
        {
            vec_trans[trans_id]->P += error_p;
            vec_trans[trans_id]->Q += error_q;
        }
    }
    new (&error_vec) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    new (&error_p) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    new (&error_q) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    
    return logsum / (1 + neg_samples) / node_cnt;
}

/*void linelib_output_batch(char *file_name, int binary, line_node **array_line_node, int array_length)
 {
 int total_node_size = 0;
 for (int k = 0; k != array_length; k++) total_node_size += array_line_node[k]->node_size;
 int vector_size = array_line_node[0]->vector_size;
 for (int k = 1; k != array_length; k++) if (array_line_node[k]->vector_size != vector_size)
 {
 printf("Error: vector dimensions are not equivalent!\n");
 exit(1);
 }
 
 FILE *fo = fopen(file_name, "wb");
 fprintf(fo, "%d %d\n", total_node_size, vector_size);
 for (int k = 0; k != array_length; k++) for (int a = 0; a != array_line_node[k]->node_size; a++)
 {
 fprintf(fo, "%s ", array_line_node[k]->node[a].word);
 if (binary) for (int b = 0; b != vector_size; b++) fwrite(&(array_line_node[k]->vec[a * vector_size + b]), sizeof(real), 1, fo);
 else for (int b = 0; b != vector_size; b++) fprintf(fo, "%lf ", array_line_node[k]->vec[a * vector_size + b]);
 fprintf(fo, "\n");
 }
 fclose(fo);
 }*/

line_trainer_path::line_trainer_path()
{
    phin = NULL;
    expTable = NULL;
    neg_samples = 0;
    dp_cnt = NULL;
    dp_cnt_fd = NULL;
    path.clear();
    path_node.clear();
    path_link.clear();
    path_size = 0;
    neg_table = NULL;
    smp_init = NULL;
    smp_init_weight = NULL;
    smp_init_index = NULL;
    smp_dp = NULL;
    smp_dp_weight = NULL;
    smp_dp_index = NULL;
}

line_trainer_path::~line_trainer_path()
{
    int node_size = phin->node_u->node_size;
    phin = NULL;
    if (expTable != NULL) {free(expTable); expTable = NULL;}
    neg_samples = 0;
    if (dp_cnt != NULL)
    {
        for (int k = 0; k != node_size; k++) if (dp_cnt[k] != NULL)
            free(dp_cnt[k]);
        free(dp_cnt);
        dp_cnt = NULL;
    }
    if (dp_cnt_fd != NULL)
    {
        for (int k = 0; k != node_size; k++) if (dp_cnt_fd[k] != NULL)
            free(dp_cnt_fd[k]);
        free(dp_cnt_fd);
        dp_cnt_fd = NULL;
    }
    path.clear();
    path_node.clear();
    path_link.clear();
    if (neg_table != NULL)
    {
        for (int k = 1; k != path_size; k++) if (neg_table[k] != NULL)
            free(neg_table[k]);
        free(neg_table);
        neg_table = NULL;
    }
    path_size = 0;
    if (smp_init != NULL)
    {
        ransampl_free(smp_init);
        smp_init = NULL;
    }
    if (smp_dp != NULL)
    {
        for (int i = 0; i != path_size; i++) for (int j = 0; j != node_size; j++) if (smp_dp[i][j] != NULL)
        {
            ransampl_free(smp_dp[i][j]);
            smp_dp[i][j] = NULL;
        }
        for (int i = 0; i != path_size; i++) {free(smp_dp[i]); smp_dp[i] = NULL;}
        free(smp_dp);
        smp_dp = NULL;
    }
}

int line_trainer_path::get_path_length()
{
    return path_size;
}

void line_trainer_path::init(std::string meta_path, line_hin *p_hin, int negative)
{
    path = meta_path;
    if (path.size() % 2 == 0)
    {
        printf("ERROR: meta-path %s error!\n", path.c_str());
        exit(1);
    }
    path_size = ((int)(path.size()) + 1) / 2;
    for (int k = 0; k != path_size; k++) path_node.append(1, path[k * 2]);
    for (int k = 0; k != path_size - 1; k++) path_link.append(1, path[k * 2 + 1]);
    phin = p_hin;
    neg_samples = negative;
    
    line_node *node_u = p_hin->node_u, *node_v = p_hin->node_v;
    if (node_u->vector_size != node_v->vector_size)
    {
        printf("ERROR: vector dimsions are not same!\n");
        exit(1);
    }
    
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    
    int node_size = node_u->node_size;
    
    dp_cnt = (double **)malloc(node_size * sizeof(double *));
    for (int k = 0; k != node_size; k++) dp_cnt[k] = NULL;
    for (int k = 0; k != node_size; k++) dp_cnt[k] = (double *)malloc(path_size * sizeof(double));
    for (int i = 0; i != node_size; i++) for (int j = 0; j != path_size; j++) dp_cnt[i][j] = 0;
    
    dp_cnt_fd = (double **)malloc(node_size * sizeof(double *));
    for (int k = 0; k != node_size; k++) dp_cnt_fd[k] = NULL;
    for (int k = 0; k != node_size; k++) dp_cnt_fd[k] = (double *)malloc(path_size * sizeof(double));
    for (int i = 0; i != node_size; i++) for (int j = 0; j != path_size; j++) dp_cnt_fd[i][j] = 0;
    
    char node_type, link_type;
    for (int step = path_size - 1; step >= 0; step--)
    {
        node_type = path_node[step];
        if (step == path_size - 1)
        {
            for (int u = 0; u != node_size; u++) if ((node_u->node[u]).type == node_type)
                dp_cnt[u][step] = 1;
        }
        else
        {
            link_type = path_link[step];
            for (int u = 0; u != node_size; u++) if ((node_u->node[u]).type == node_type)
            {
                int neighbor_size = (int)(phin->hin[u].size());
                for (int i = 0; i != neighbor_size; i++)
                {
                    int v = phin->hin[u][i].nb_id;
                    char cur_link_type = phin->hin[u][i].eg_tp;
                    double wei = phin->hin[u][i].eg_wei;
                    if ((node_u->node[v]).type == path_node[step + 1] && link_type == cur_link_type)
                        dp_cnt[u][step] += dp_cnt[v][step + 1] * wei;
                }
            }
        }
    }
    
    for (int step = 0; step < path_size - 1; step++)
    {
        char node_type = path_node[step];
        char link_type = path_link[step];
        for (int u = 0; u != node_size; u++) if ((node_u->node[u]).type == node_type)
        {
            if (step == 0) dp_cnt_fd[u][step] = 1;
            
            int neighbor_size = (int)(phin->hin[u].size());
            for (int i = 0; i != neighbor_size; i++)
            {
                int v = phin->hin[u][i].nb_id;
                char cur_link_type = phin->hin[u][i].eg_tp;
                double wei = phin->hin[u][i].eg_wei;
                if ((node_u->node[v]).type == path_node[step + 1] && link_type == cur_link_type)
                    dp_cnt_fd[v][step + 1] += dp_cnt_fd[u][step] * wei;
            }
        }
    }
    
    // Init negative sampling table
    neg_table = (int **)malloc(path_size * sizeof(int *));
    for (int k = 1; k != path_size; k++) neg_table[k] = NULL;
    for (int k = 1; k != path_size; k++) neg_table[k] = (int *)malloc(neg_table_size * sizeof(int));
    for (int k = 1; k != path_size; k++)
    {
        int a, i;
        double total_pow = 0, d1;
        double power = 1;//0.75;
        for (a = 0; a < node_size; a++) total_pow += pow(dp_cnt_fd[a][k], power);
        a = 0; i = 0;
        d1 = pow(dp_cnt_fd[i][k], power) / (double)total_pow;
        while (a < neg_table_size) {
            if ((a + 1) / (double)neg_table_size > d1) {
                i++;
                if (i >= node_size) {i = node_size - 1; d1 = 2;}
                d1 += pow(dp_cnt_fd[i][k], power) / (double)total_pow;
            }
            else
                neg_table[k][a++] = i;
        }
    }
    
    int node_cnt;
    
    // Init the sampling table of step 0
    node_cnt = 0;
    node_type = path_node[0];
    for (int u = 0; u != node_size; u++) if ((node_u->node[u]).type == node_type)
        node_cnt++;
    smp_init_index = (int *)malloc(node_cnt * sizeof(int));
    smp_init_weight = (double *)malloc(node_cnt * sizeof(double));
    smp_init = ransampl_alloc(node_cnt);
    node_cnt = 0;
    for (int u = 0; u != node_size; u++) if ((node_u->node[u]).type == node_type)
    {
        smp_init_index[node_cnt] = u;
        smp_init_weight[node_cnt] = dp_cnt[u][0];
        node_cnt++;
    }
    ransampl_set(smp_init, smp_init_weight);
    
    // Init sampling tables of the following steps
    smp_dp_index = (int ***)malloc(path_size * sizeof(int **));
    for (int k = 0; k != path_size; k++) smp_dp_index[k] = (int **)malloc(node_size * sizeof(int *));
    for (int i = 0; i != path_size; i++) for (int j = 0; j != node_size; j++) smp_dp_index[i][j] = NULL;
    smp_dp_weight = (double ***)malloc(path_size * sizeof(double **));
    for (int k = 0; k != path_size; k++) smp_dp_weight[k] = (double **)malloc(node_size * sizeof(double *));
    for (int i = 0; i != path_size; i++) for (int j = 0; j != node_size; j++) smp_dp_weight[i][j] = NULL;
    smp_dp = (ransampl_ws ***)malloc(path_size * sizeof(ransampl_ws));
    for (int k = 0; k != path_size; k++) smp_dp[k] = (ransampl_ws **)malloc(node_size * sizeof(ransampl_ws));
    for (int i = 0; i != path_size; i++) for (int j = 0; j != node_size; j++) smp_dp[i][j] = NULL;
    
    for (int step = 0; step != path_size - 1; step++)
    {
        node_type = path_node[step];
        link_type = path_link[step];
        for (int u = 0; u != node_size; u++) if((node_u->node[u]).type == node_type)
        {
            node_cnt = 0;
            int neighbor_size = (int)(phin->hin[u].size());
            for (int i = 0; i != neighbor_size; i++)
            {
                int v = phin->hin[u][i].nb_id;
                char cur_edge_tp = phin->hin[u][i].eg_tp;
                if ((node_u->node[v]).type == path_node[step + 1] && cur_edge_tp == link_type)
                    node_cnt++;
            }
            if (node_cnt == 0) continue;
            
            smp_dp_index[step][u] = (int *)malloc(node_cnt * sizeof(int));
            smp_dp_weight[step][u] = (double *)malloc(node_cnt * sizeof(double));
            smp_dp[step][u] = ransampl_alloc(node_cnt);
            node_cnt = 0;
            for (int i = 0; i != neighbor_size; i++)
            {
                int v = phin->hin[u][i].nb_id;
                char cur_edge_tp = phin->hin[u][i].eg_tp;
                double wei = phin->hin[u][i].eg_wei;
                if ((node_u->node[v]).type == path_node[step + 1] && cur_edge_tp == link_type)
                {
                    smp_dp_index[step][u][node_cnt] = v;
                    smp_dp_weight[step][u][node_cnt] = dp_cnt[v][step + 1] * wei;
                    node_cnt++;
                }
            }
            ransampl_set(smp_dp[step][u], smp_dp_weight[step][u]);
        }
    }
    
    // add trans
    for (int i = 0; i < path_size; i++) for (int j = i + 1; j < path_size; j++)
    {
        std::string tp = path.substr(i * 2, (j - i) * 2 + 1);
        if (line_trainer_path::map_trans[tp] == 0)
        {
            line_trans *ptrans = new line_trans;
            ptrans->init(tp, phin->node_u->vector_size);
            line_trainer_path::vec_trans.push_back(ptrans);
            line_trainer_path::map_trans[tp] = line_trainer_path::cnt_trans + 1;
            line_trainer_path::cnt_trans++;
        }
    }
}

void line_trainer_path::print_trans()
{
    std::map<std::string, int>::iterator iter;
    for (iter = line_trainer_path::map_trans.begin(); iter != line_trainer_path::map_trans.end(); iter++)
    {
        std::string tp = iter->first;
        int id = iter->second;
        printf("%s %d %s\n", tp.c_str(), id, line_trainer_path::vec_trans[id - 1]->type.c_str());
    }
}

void line_trainer_path::sample_path(int *node_lst, double (*func_rand_num)())
{
    long long cur_entry;
    // Sample the first node
    cur_entry = ransampl_draw(smp_init, func_rand_num(), func_rand_num());
    node_lst[0] = smp_init_index[cur_entry];
    // Sample following nodes
    for (int step = 0; step != path_size - 1; step++)
    {
        int u = node_lst[step];
        cur_entry = ransampl_draw(smp_dp[step][u], func_rand_num(), func_rand_num());
        node_lst[step + 1] = smp_dp_index[step][u][cur_entry];
    }
}

double line_trainer_path::train_path(int mode, int *node_lst, real alpha, real *_error_vec, real *_error_p, real *_error_q, double (*func_rand_num)(), unsigned long long &rand_index, int model)
{
    double logsum = 0;
    int cnt = 0;
    
    int target, label, u, v, vector_size, trans_id;
    real f, g;
    std::string path_type = std::string();
    int step_bg = 0, step_ed = 0, pst_bg = 0, pst_ed = 0;
    line_node *node_u = phin->node_u, *node_v = phin->node_v;
    
    vector_size = node_u->vector_size;
    Eigen::Map<BLPVector> error_vec(_error_vec, vector_size);
    Eigen::Map<BLPVector> error_p(_error_p, vector_size);
    Eigen::Map<BLPVector> error_q(_error_q, vector_size);
    error_vec.setZero();
    error_p.setZero();
    error_q.setZero();
    
    sample_path(node_lst, func_rand_num);
    
    if (model == 0)
    {
        step_bg = path_size - 1; step_ed = path_size;
        pst_bg = 0; pst_ed = 1;
    }
    else if (model == 1)
    {
        step_bg = 1; step_ed = 2;
        pst_bg = 0; pst_ed = path_size - 1;
    }
    else if (model == 2)
    {
        step_bg = 1; step_ed = path_size;
        pst_bg = 0; pst_ed = path_size - 1;
    }
    
    for (int step = step_bg; step != step_ed; step++) for (int pst = pst_bg; pst != pst_ed; pst++)
    {
        if (pst + step >= path_size) continue;
        
        path_type = path.substr(pst * 2, step * 2 + 1);
        trans_id = map_trans[path_type];
        if (trans_id == 0)
        {
            printf("ERROR: trans id error!\n");
            exit(1);
        }
        trans_id--;
        
        u = node_lst[pst];
        v = node_lst[pst + step];
        
        error_vec.setZero();
        error_p.setZero();
        error_q.setZero();
        
        for (int d = 0; d < neg_samples + 1; d++)
        {
            if (d == 0)
            {
                target = v;
                label = 1;
            }
            else
            {
                rand_index = rand_index * (unsigned long long)25214903917 + 11;
                //target = neg_table[pst + step][(rand_index >> 16) % neg_table_size];
                target = neg_table[pst + step][(rand_index >> 16) % neg_table_size];
                if (target == v) continue;
                label = 0;
            }
            f = 0;
            f += node_u->vec.row(u) * node_v->vec.row(target).transpose();
            if (mode == 1 || mode == 2 || mode == 3)
            {
                f += (node_u->vec.row(u)) * (vec_trans[trans_id]->P.transpose());
                f += (node_v->vec.row(target)) * (vec_trans[trans_id]->Q.transpose());
                f += vec_trans[trans_id]->bias;
            }
            
            if (label == 1) logsum += log(sigmoid(f));
            else logsum += log(1 - sigmoid(f));
            cnt += 1;
            
            if (f > MAX_EXP) g = (label - 1) * alpha;
            else if (f < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            if (mode == 2 || mode == 3)
            {
                error_p += g * (node_u->vec.row(u));
                error_q += g * (node_v->vec.row(target));
            }
            if (mode == 1 || mode == 3)
            {
                error_vec += g * ((node_v->vec.row(target)));
                error_vec += g * (vec_trans[trans_id]->P);
                
                node_v->vec.row(target) += g * ((node_u->vec.row(u)));
                node_v->vec.row(target) += g * (vec_trans[trans_id]->Q);
            }
            if (mode == 0)
            {
                error_vec += g * node_v->vec.row(target);
                node_v->vec.row(target) += g * node_u->vec.row(u);
            }
        }
        if (mode == 0 || mode == 1 || mode == 3) node_u->vec.row(u) += error_vec;
        if (mode == 2 || mode == 3)
        {
            vec_trans[trans_id]->P += error_p;
            vec_trans[trans_id]->Q += error_q;
        }
    }
    new (&error_vec) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    new (&error_p) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    new (&error_q) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    
    return logsum / cnt;
}
