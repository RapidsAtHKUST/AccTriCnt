#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <set>
#include <functional>
#include <climits>
#include <ctime>
#include <stdlib.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <sys/time.h>

#include "Graph.h"
#include "Util.h"


#include "../../../utils/yche_serialization.h"

using namespace std;

const int INPUTNUM = 1;


int main(int argc, char *argv[]) {
    ios::sync_with_stdio(false);
    int i;
    int W = 5;
    clock_t start, end;
    string filename;

    if (argc == 1) {
        cout << "please provide parameter" << endl;
        exit(0);
    }

    i = 1;
    while (i < argc) {
        if (strcmp("-w", argv[i]) == 0) {
            i++;
            W = atoi(argv[i]);
            if (W <= 0) {
                cout << "w should be larger than 0" << endl;
                quit();
            }
            i++;
        } else {
            filename = argv[i++];
        }
    }

    srand(time(0));

    Graph g;
    string name;
    name = extractFilename(filename.c_str());
    g.setFilename(name);

    start = clock();
    g.readGraph(filename);

#ifdef RCM
    vector<int> rcm_order;
    g.Transform(rcm_order);
#endif
    cout << name << " readGraph is complete." << endl;
    end = clock();
    cout << "Time Cost: " << (double) (end - start) / CLOCKS_PER_SEC << endl;

    start = clock();
    vector<int> cache_order;
    g.GorderGreedy(cache_order, W);
    end = clock();
    cout << "ReOrdered Time Cost: " << (double) (end - start) / CLOCKS_PER_SEC << endl;
//	cout << "Begin Output the Reordered Graph" << endl;
//    g.PrintReOrderedGraph(cache_order);
//	cout << endl;

    /* Flush To .dict*/
#ifdef RCM
    string file_path = filename + "/" + "rcm-cache.dict";
#else
    string file_path = filename + "/" + "cache.dict";
#endif

    FILE *pFile = fopen(file_path.c_str(), "wb");
    YcheSerializer serializer;

#ifdef RCM
    vector<int> real_order = rcm_order;
    for (auto i = 0; i < real_order.size(); i++) {
        real_order[i] = cache_order[rcm_order[i]];
    }
    serializer.write_vec(pFile, real_order);
#else
    serializer.write_vec(pFile, cache_order);
#endif
// flush and close the file handle
    fflush(pFile);
    fclose(pFile);

}

