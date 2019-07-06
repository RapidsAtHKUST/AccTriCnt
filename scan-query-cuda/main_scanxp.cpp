//
// Created by yche on 10/31/17.
//
#include <chrono>

#include "scan_xp.h"
#include "util/log.h"

void Usage() {
    cout << "Usage: [1]exe [2]graph-dir [3]similarity-threshold "
            "[4]density-threshold [5 thread_num] [6 optional]output\n";
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
    log_f = fopen(argv[5], "a+");
    log_set_fp(log_f);
#endif

    // parse parameters
#ifdef GIT_SHA1
    log_info("git version: %s", GIT_SHA1);
#endif
    log_info("graph dir: %s", argv[1]);
    SCAN_XP scanxp(NUMT, MY_U, EPSILON, argv[1]);
    scanxp.Execute();
}
