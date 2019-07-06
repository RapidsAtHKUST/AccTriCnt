//
// Created by yche on 10/31/17.
//
#include <chrono>

#include "scan_xp.h"
#include "util/log.h"


string reorder_method;
string dir;

void Usage() {
    cout << "Usage: [1]exe [2]graph-dir [3]similarity-threshold "
            "[4]density-threshold [5 thread_num] [6 reorder_method]output\n";
}

int main(int argc, char *argv[]) {
    using namespace chrono;
    if (argc < 4) {
        Usage();
        return 0;
    }


    auto EPSILON = strtod(argv[2], nullptr);
    auto MY_U = atoi(argv[3]);
    auto NUMT = atoi(argv[4]);

    //set log file descriptor
#ifdef USE_LOG
    FILE *log_f;
    log_f = fopen(argv[6], "a+");
    log_set_fp(log_f);
#endif

    dir = argv[1];
    reorder_method = argv[5];
    // parse parameters
    log_info("git version: %s", GIT_SHA1);
    log_info("graph dir: %s", argv[1]);
    SCAN_XP scanxp(NUMT, MY_U, EPSILON, argv[1]);
    scanxp.Execute();
}
