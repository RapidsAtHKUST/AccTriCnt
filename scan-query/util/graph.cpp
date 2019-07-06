#include"graph.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstring>

#ifdef MEM_KIND

#include <hbwmalloc.h>

#endif

#include <chrono>
#include <algorithm>
#include <fcntl.h>

#ifdef TBB

#include <tbb/parallel_sort.h>
//#include <tbb/parallel_scan.h>

#endif

#include "log.h"
#include "util.h"
#include "graph.h"


using namespace chrono;

Graph::Graph(char *dir_cstr) {
    dir = string(dir_cstr);

    ReadDegree();
    ReadAdjacencyList();
    CheckInputGraph();

    // vertex property
    label = new int[nodemax];
    core_count = new int[nodemax];

    // edge property
    common_node_num = new int[edgemax];
    similarity = new bool[edgemax];
}

void Graph::ReadDegree() {
    auto start = high_resolution_clock::now();

    ifstream deg_file(dir + string("/b_degree.bin"), ios::binary);
    int int_size;
    deg_file.read(reinterpret_cast<char *>(&int_size), 4);

    deg_file.read(reinterpret_cast<char *>(&nodemax), 4);
    deg_file.read(reinterpret_cast<char *>(&edgemax), 4);
    log_info("int size: %d, n: %s, m: %s", int_size, FormatWithCommas(nodemax).c_str(),
             FormatWithCommas(edgemax).c_str());

    degree.resize(static_cast<unsigned long>(nodemax));
    deg_file.read(reinterpret_cast<char *>(&degree.front()), sizeof(int) * nodemax);

    auto end = high_resolution_clock::now();
    log_info("read degree file time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);
}

void Graph::ReadAdjacencyList() {
    auto start = high_resolution_clock::now();
    ifstream adj_file(dir + string("/b_adj.bin"), ios::binary);

    // csr representation
    node_off = new uint32_t[nodemax + 1];
#ifdef HUGE_PAGE
    const char *FILE_NAME = "/mnt/huge/edge.tmp";
    int fd = open(FILE_NAME, O_CREAT | O_RDWR, 0755);
    if (fd < 0) {
        log_error("Open failed");
        unlink(FILE_NAME);
        exit(1);
    }
    edge_dst = (int *) mmap(0, static_cast<uint64_t >(edgemax + 16) * 4u, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (edge_dst == MAP_FAILED) {
        log_error("mmap");
        exit(1);
    }
    log_info("Returned address is %p\n", edge_dst);

#elif defined(MEM_KIND)
    //    edge_dst = new int[edgemax + 16];   // padding for simd
    // allocation on high-bandwidth memory (16GB)
    edge_dst = static_cast<int *>(hbw_malloc(sizeof(int) * static_cast<uint64_t>(edgemax + 16)));
#else
    edge_dst = static_cast<int *>(malloc(sizeof(int) * static_cast<uint64_t>(edgemax + 16)));
#endif

    string dst_v_file_name = dir + string("/b_adj.bin");
    auto dst_v_fd = open(dst_v_file_name.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
    int *buffer = (int *) mmap(0, static_cast<uint64_t >(edgemax) * 4u, PROT_READ, MAP_PRIVATE, dst_v_fd, 0);

    // prefix sum
    node_off[0] = 0;
    for (auto i = 0; i < nodemax; i++) { node_off[i + 1] = node_off[i] + degree[i]; }

    auto end = high_resolution_clock::now();
    log_info("malloc, and sequential-scan time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);
    // load dst vertices into the array
#pragma omp parallel for schedule(dynamic, 1000)
    for (auto i = 0; i < nodemax; i++) {
        // copy to the high memory bandwidth mem
        for (uint64_t offset = node_off[i]; offset < node_off[i + 1]; offset++) {
            edge_dst[offset] = buffer[offset];
        }
        // inclusive
        degree[i]++;
    }
    munmap(buffer, static_cast<uint64_t >(edgemax) * 4u);

    auto end2 = high_resolution_clock::now();
    log_info("read adjacency list file time: %.3lf s", duration_cast<milliseconds>(end2 - end).count() / 1000.0);
}

void Graph::CheckInputGraph() {
    auto start = high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 5000)
    for (auto i = 0; i < nodemax; i++) {
        for (auto j = node_off[i]; j < node_off[i + 1]; j++) {
            if (edge_dst[j] == i) {
                cout << "Self loop\n";
                exit(1);
            }
            if (j > node_off[i] && edge_dst[j] <= edge_dst[j - 1]) {
                log_error("Edges not sorted in increasing id order!\nThe program may not run properly!");
                log_error("node_off[%d]: %zu, %zu; cur j: %zu; (%d, %d)", i, node_off[i], node_off[i + 1], j,
                          edge_dst[j],
                          edge_dst[j - 1]);
                exit(1);
            }
        }
    }
    auto end = high_resolution_clock::now();
    log_info("check input graph file time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);
}

void Graph::Output(const char *eps_s, const char *min_u, UnionFind *union_find_ptr) {
    string out_name = dir + "/scanxp-result-" + string(eps_s) + "-" + string(min_u) + ".txt";
    ofstream ofs(out_name);
    ofs << "c/n vertex_id cluster_id\n";

    // observation 2: unique belonging
    auto start = high_resolution_clock::now();
    for (auto i = 0; i < nodemax; i++) {
        if (label[i] == CORE) {
            ofs << "c " << i << " " << cluster_dict[union_find_ptr->FindRoot(i)] << "\n";
        }
    }
    auto end = high_resolution_clock::now();
    log_info("cores output time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);

    // possibly multiple belongings
#ifdef TBB
    tbb::parallel_sort(noncore_cluster.begin(), noncore_cluster.end());
    auto tmp = high_resolution_clock::now();
    log_info("parallel sort time: %.3lf s", duration_cast<milliseconds>(tmp - end).count() / 1000.0);
#else
    sort(noncore_cluster.begin(), noncore_cluster.end());
#endif

    auto iter_end = unique(noncore_cluster.begin(), noncore_cluster.end());
    for_each(noncore_cluster.begin(), iter_end,
             [&ofs](pair<int, int> my_pair) { ofs << "n " << my_pair.second << " " << my_pair.first << "\n"; });
    auto end2 = high_resolution_clock::now();
    log_info("non-cores output time: %.3lf s", duration_cast<milliseconds>(end2 - end).count() / 1000.0);
}

void Graph::Output(const char *eps_s, const char *min_u, UnionFind *union_find_ptr,
                   vector<int> &old_vid_dict) {
    string out_name = dir + "/scanxp-result-" + string(eps_s) + "-" + string(min_u) + ".txt";
    ofstream ofs(out_name);
    ofs << "c/n vertex_id cluster_id\n";

    // observation 2: unique belonging
    vector<pair<int, int>> pairs;
    auto start = high_resolution_clock::now();
    for (auto i = 0; i < nodemax; i++) {
        if (label[i] == CORE) {
            pairs.emplace_back(old_vid_dict[i], cluster_dict[union_find_ptr->FindRoot(i)]);
        }
    }
    sort(begin(pairs), end(pairs), [](pair<int, int> l, pair<int, int> r) {
        return l.first < r.first;
    });
    for (auto p:pairs) {
        ofs << "c " << p.first << " " << p.second << "\n";
    }

    auto end = high_resolution_clock::now();
    log_info("cores output time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);

    // possibly multiple belongings
#ifdef TBB
    tbb::parallel_sort(noncore_cluster.begin(), noncore_cluster.end());
    auto tmp = high_resolution_clock::now();
    log_info("parallel sort time: %.3lf s", duration_cast<milliseconds>(tmp - end).count() / 1000.0);
#else
    sort(noncore_cluster.begin(), noncore_cluster.end());
#endif

    auto iter_end = unique(noncore_cluster.begin(), noncore_cluster.end());
    for_each(noncore_cluster.begin(), iter_end,
             [&ofs](pair<int, int> my_pair) {
                 ofs << "n " << my_pair.second << " " << my_pair.first << "\n";
             });
    auto end2 = high_resolution_clock::now();
    log_info("non-cores output time: %.3lf s", duration_cast<milliseconds>(end2 - end).count() / 1000.0);
}

