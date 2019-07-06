#ifndef SCAN_QUERY_SCANXP
#define SCAN_QUERY_SCANXP

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <omp.h> //OpenMP

#include <iostream>
#include <vector>
#include <fstream>
#include <set>
#include <atomic> //CAS

#include "set-inter/csr_set_intersection_serial.h"

using namespace std;

class SCAN_XP {
private:
    int min_u_;
    double epsilon_;

private:
    Graph g;
    UnionFind *uf_ptr;

private:
    bool CheckHub(Graph *g, UnionFind *uf, int a);

    uint32_t BinarySearch(int *array, uint32_t offset_beg, uint32_t offset_end, int val);

private:
    void CheckCore(Graph *g);

    void ClusterCore();

    void LabelNonCore();

    void PostProcess();

    void MarkClusterMinEleAsId(UnionFind *union_find_ptr);

    void PrepareResultOutput();

public:
    int thread_num_;

    int FindSrc(Graph *g, int u,  uint32_t edge_idx);

public:
    SCAN_XP(int thread_num, int min_u, double epsilon, char *dir);

    ~SCAN_XP();

    void Execute();
};

#endif

