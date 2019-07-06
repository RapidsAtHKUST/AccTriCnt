#ifndef __STAT_H__
#define __STAT_H__

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "graph.h"
#include "log.h"

extern int parseLine(char *line);

extern int getValue();

struct LowerBoundStat {
    uint32_t cnt_;
    uint64_t acc_time_;
    long min_time_;

    LowerBoundStat() {
        cnt_ = 0;
        acc_time_ = 0;
        min_time_ = INT32_MAX;
    }

    void MergeStat(const LowerBoundStat &other) {
        cnt_ += other.cnt_;
        acc_time_ += other.acc_time_;
        min_time_ = min(min_time_, other.min_time_);
    }

    void AddStat(long time, Graph *g, uint32_t i, int u, int v) {
        cnt_++;
        acc_time_ += time;
        if (time == 0) {
            log_info("trivial");
        }
        min_time_ = min(min_time_, time);
    }
};

struct InterSectStat {
    uint32_t cnt_;
    uint64_t acc_time_;

    long min_time_;
    long max_time_;

    double acc_select_ratio_;

    uint64_t acc_select_;
    uint32_t min_select_;
    uint32_t max_select_;

    uint64_t acc_min_degree_;
    uint64_t acc_max_degree_;

    uint64_t acc_skew_;
    uint32_t min_skew_;
    uint32_t max_skew_;

    InterSectStat() {
        cnt_ = 0;
        acc_time_ = 0;

        acc_select_ = 0;
        acc_skew_ = 0;

        min_select_ = UINT32_MAX;
        max_select_ = 0;
        min_skew_ = UINT32_MAX;
        max_skew_ = 0;

        acc_select_ratio_ = 0;
        acc_min_degree_ = 0;
        acc_max_degree_ = 0;

        min_time_ = INT32_MAX;
        max_time_ = 0;
    }

    void MergeStat(const InterSectStat &other) {
        cnt_ += other.cnt_;
        acc_time_ += other.acc_time_;
        acc_select_ += other.acc_select_;
        acc_skew_ += other.acc_skew_;
        min_select_ = min<uint32_t>(min_select_, other.min_select_);
        max_select_ = max<uint32_t>(max_select_, other.max_select_);
        min_skew_ = min<uint32_t>(min_skew_, other.min_skew_);
        max_skew_ = max<uint32_t>(max_skew_, other.max_skew_);

        acc_select_ratio_ += other.acc_select_ratio_;
        acc_min_degree_ += other.acc_min_degree_;
        acc_max_degree_ += other.acc_max_degree_;

        min_time_ = min(min_time_, other.min_time_);
        max_time_ = max(max_time_, other.max_time_);
    }

    void AddStat(long time, Graph *g, uint32_t i, int u, int v) {
        cnt_++;
        acc_time_ += time;

        acc_select_ratio_ += static_cast<double>(g->common_node_num[i] - 2) / min(g->degree[u], g->degree[v]);
        acc_select_ += g->common_node_num[i] - 2;
        min_select_ = min<uint32_t>(min_select_, g->common_node_num[i] - 2);
        max_select_ = max<uint32_t>(max_select_, g->common_node_num[i] - 2);
        auto skew = max(g->degree[u], g->degree[v]) / min(g->degree[u], g->degree[v]);

        acc_skew_ += skew;
        min_skew_ = min<uint32_t>(min_skew_, skew);
        max_skew_ = max<uint32_t>(max_skew_, skew);

        acc_min_degree_ += min(g->degree[u], g->degree[v]);
        acc_max_degree_ += max(g->degree[u], g->degree[v]);

        min_time_ = min(min_time_, time);
        max_time_ = max(max_time_, time);
    }
};

#endif

