//
// Created by yche on 2/28/19.
//

#include <chrono>

#include "scan_xp.h"
#include "util/log.h"

string reorder_method;
string dir;

int main(int argc, char *argv[]) {
    using namespace chrono;

    auto NUMT = atoi(argv[2]);
    //set log file descriptor
#ifdef USE_LOG
    FILE *log_f;
    log_f = fopen(argv[4], "a+");
    log_set_fp(log_f);
#endif

    // parse parameters
    log_info("git version: %s", GIT_SHA1);
    log_info("graph dir: %s", argv[1]);
    dir = string(argv[1]);

    SCAN_XP scan_xp(NUMT, argv[1]);
    reorder_method = string(argv[3]);
    scan_xp.TriCnt();
}
