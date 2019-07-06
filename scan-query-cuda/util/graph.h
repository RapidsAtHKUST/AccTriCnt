#ifndef SCAN_QUERY_GRAPH
#define SCAN_QUERY_GRAPH

#include<cstdlib>
#include<ctime>

#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<sstream>

#include "union_find.h"

using namespace std;

#define UNCLASSIFIED 0
#define CORE 1 //Member of cluster
#define HUB 2
//#define BALANCE 8192

struct Graph {
    uint32_t nodemax;
    uint32_t edgemax;

    // csr representation
    uint32_t *node_off;
    int *edge_dst;

    // balanced offset array by splitting single vertex into multiple balanced ones

    // edge property
    bool *similarity;
    int *common_node_num;

    // vertex property
    int *core_count;
    int *label;

    // other
    vector<int> degree;

    // clusters: core and non-core(hubs)
    vector<int> cluster_dict;    // observation 2: core vertex clusters are disjoint

    // first: cluster id(min core-vertex id in cluster), second: non-core vertex id
    vector<pair<int, int>> noncore_cluster; // observation 1: clusters may overlap, observation 3: non-core uniquely determined by core

    string dir;

    explicit Graph(char *dir_cstr);

public:
    void ReadDegree();

    void CheckInputGraph();

    void ReadAdjacencyList();

    void Output(const char *eps_s, const char *min_u, UnionFind *union_find_ptr);
};

#endif