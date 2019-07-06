//
// Created by yche on 8/26/18.
//

#include <chrono>
#include <cassert>

#include "../utils/log.h"
#include "../utils/yche_serialization.h"

#include "pscan_graph.h"


vector<pair<int, int>> GetEdgeList(string input_file_path, int &max_ele) {
    vector<pair<int, int>> lines;

    ifstream ifs(input_file_path);

    while (ifs.good()) {
        string tmp_str;
        stringstream ss;
        std::getline(ifs, tmp_str);
        if (!ifs.good())
            break;
        if (tmp_str[0] != '#') {
            ss.clear();
            ss << tmp_str;
            int first, second;
            ss >> first >> second;
            // 1st case first == second: skip these self loop, (i,i)
            // 2nd case first > second: unique (i,j), (j,i)
            if (first >= second) {
                continue;
            }
            assert(first < INT32_MAX and second < INT32_MAX);
            if (second > max_ele)
                max_ele = second;
            lines.emplace_back(first, second);
        }
    }
    return lines;
}

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[]) {
    //set log file descriptor
#ifdef USE_LOG
    FILE *log_f;
    if (argc >= 3) {
        log_f = fopen(argv[2], "a+");
        log_set_fp(log_f);
    }
#endif


    auto start = high_resolution_clock::now();
    int tmp;
    auto edges = GetEdgeList(string(argv[1]) + "/edge_lst.txt", tmp);
    auto end = high_resolution_clock::now();
    log_info("construct time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);



    // 2nd: output
    string my_path = string(argv[1]) + "/" + "undir_edge_list.bin";
    FILE *pFile = fopen(my_path.c_str(), "wb");
    YcheSerializer serializer;
    serializer.write_array(pFile, &edges.front(), edges.size());

    // flush and close the file handle
    fflush(pFile);
    fclose(pFile);

    auto end2 = high_resolution_clock::now();
    log_info("output time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);

#ifdef USE_LOG
    fflush(log_f);
    fclose(log_f);
#endif
}