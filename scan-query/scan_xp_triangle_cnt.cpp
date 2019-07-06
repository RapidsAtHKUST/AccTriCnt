#include "scan_xp.h"

#include "util/fake_header.h"

#include <chrono>
#include <vector>
#include <unordered_set>

#ifdef HBW

#include <memkind.h>
#include <hbw_allocator.h>

#endif


#include <sparsepp/spp.h>

#include "util/stat.h"
#include "util/util.h"
#include "util/yche_serialization.h"


#ifdef ROARING

#include "roaring.hh"

#endif

#include "libpopcnt.h"

#include "set-inter/lemire/intersection.h"
#include "set-inter/han/intersection_algos.hpp"

#ifdef TETZANK

#include "set-inter/tetzank/intersection/avx.hpp"

#endif

#if defined(TETZANK) && defined(__AVX2__)

#include "set-inter/tetzank/intersection/avx2.hpp"

#endif

#ifdef EMPTY_HEADED

#include "emptyheaded.hpp"

#endif

using namespace std;
using namespace std::chrono;

SCAN_XP::SCAN_XP(int thread_num, char *dir) : thread_num_(thread_num), g(Graph(dir)), uf_ptr(nullptr) {
}

extern string reorder_method;
extern string dir;

void SCAN_XP::TriCnt() {
    auto clk_beg = high_resolution_clock::now();

    if (reorder_method == "deg") {
        // Counting on the Degree-Ordered-Directed-Graph (View).
        log_info("Re-ordering... Method: DODG");
        ReorderDegDescending(g);
    } else if (reorder_method == "kcore") {
        log_info("Re-ordering... Method: Kcore");
        ReorderKCoreDegeneracy(g);
    } else if (reorder_method == "random") {
        log_info("Re-ordering... Method: Random");
        ReorderRandom(g);
    } else if (reorder_method == "gro" || reorder_method == "cache" ||
               reorder_method == "bfsr" || reorder_method == "dfs" ||
               reorder_method == "hybrid" || reorder_method == "rcm-cache"
               || reorder_method == "slashburn") {
        log_info("Re-ordering... Method: %s", reorder_method.c_str());
        string reorder_file_path = dir + "/" + reorder_method + ".dict";
        FILE *pFile = fopen(reorder_file_path.c_str(), "r");
        YcheSerializer serializer;
        serializer.read_array(pFile, new_vid_dict);
        fclose(pFile);

        old_vid_dict = vector<int>(g.degree.size());
        assert(new_vid_dict.size() == old_vid_dict.size());
        for (auto v = 0; v < g.nodemax; v++) {
            old_vid_dict[new_vid_dict[v]] = v;
        }
        Reorder(g);
    } else {
        log_info("Preserving the original order...");
    }
    auto clk_end = high_resolution_clock::now();
    auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
    log_info("PreProcess Reordering Cost: %.9lf s, %s KB", time / pow(10, 9), FormatWithCommas(getValue()).c_str());

#if defined(HASH)
    TriCntHash();
#elif defined(BIT_VEC)
    TriCntBitmap();
#elif defined(BIT_VEC_ADV)
    TriCntBitmapAdv();
#elif defined(BIT_VEC_OP)
    TriCntBitmapOp();
#elif defined(EMPTY_HEADED)
    TriCntEmptyHeaded();
#elif defined(ROARING)
    TriCntRoaring();
#elif defined(BSR)
    TriCntBSR();
#else
    TriCntSortedArray();
#endif
}

void SCAN_XP::TriCntBSR() {
#ifdef BSR
    log_info("Before BSRSet... %s KB", FormatWithCommas(getValue()).c_str());
    vector<BSRSet> bsrs(g.nodemax);


    auto max_d = 0;
    auto *node_off_end = (uint32_t *) malloc(sizeof(uint32_t) * (g.nodemax + 1));

#pragma omp parallel num_threads(thread_num_)
    {
#pragma omp for reduction(max: max_d)
        for (auto u = 0u; u < g.nodemax; u++) {
            node_off_end[u + 1] = static_cast<uint32_t>(
                    lower_bound(g.edge_dst + g.node_off[u], g.edge_dst + g.node_off[u + 1], u) - g.edge_dst);
            max_d = max<int>(max_d, node_off_end[u + 1] - g.node_off[u]);
        }
        int *tmp_base = new int[max_d];
        int *tmp_state = new int[max_d];


#pragma omp single
        log_info("finish init node_off_end, max d: %d", max_d);

        auto clk_beg = high_resolution_clock::now();
#pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < g.nodemax; i++) {

            auto degree = node_off_end[i + 1] - g.node_off[i];
            auto tmp_size = offline_uint_trans_bsr(g.edge_dst + g.node_off[i], degree, tmp_base, tmp_state);

            bsrs[i].base_ = new int[degree];
            bsrs[i].states_ = new int[degree];
            bsrs[i].size_ = tmp_size;
            memcpy(bsrs[i].base_, tmp_base, static_cast<size_t>(tmp_size) * sizeof(int));
            memcpy(bsrs[i].states_, tmp_state, static_cast<size_t>(tmp_size) * sizeof(int));
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("PreProcess-BSRSet Cost: %.9lf s, %s KB", time / pow(10, 9), FormatWithCommas(getValue()).c_str());
        }
    }

    uint64_t triangle_cnt = 0;
#pragma omp parallel num_threads(thread_num_)
    {
        auto start = high_resolution_clock::now();
        int *tmp_base = new int[max_d];
        int *tmp_state = new int[max_d];
#pragma omp for schedule(dynamic, 100) reduction(+:triangle_cnt)
        for (auto u = 0u; u < g.nodemax; u++) {
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
#if defined(SCALAR_MERGE_BSR)
                auto size_c = intersect_scalarmerge_bsr(bsrs[u].base_, bsrs[u].states_, bsrs[u].size_,
                                                        bsrs[v].base_, bsrs[v].states_, bsrs[v].size_,
                                                        tmp_base, tmp_state);
#elif defined(SCALAR_GALLOPING_BSR)
                auto size_c = intersect_scalargalloping_bsr(bsrs[u].base_, bsrs[u].states_, bsrs[u].size_,
                        bsrs[v].base_, bsrs[v].states_, bsrs[v].size_,tmp_base, tmp_state);
#elif defined(SIMD_GALLOPING_BSR)
                auto size_c = intersect_simdgalloping_bsr(bsrs[u].base_, bsrs[u].states_, bsrs[u].size_,
                                                          bsrs[v].base_, bsrs[v].states_, bsrs[v].size_,
                                                          tmp_base, tmp_state);
#elif defined(SHUFFLE_BSR)
                auto size_c = intersect_shuffle_bsr_b4(bsrs[u].base_, bsrs[u].states_, bsrs[u].size_,
                                                       bsrs[v].base_, bsrs[v].states_, bsrs[v].size_,
                                                       tmp_base, tmp_state);
#else
                auto size_c = intersect_qfilter_bsr_b4_v2(bsrs[u].base_, bsrs[u].states_, bsrs[u].size_,
                                                          bsrs[v].base_, bsrs[v].states_, bsrs[v].size_,
                                                          tmp_base, tmp_state);
#endif

                // popcnt here.
                triangle_cnt += popcnt(tmp_state, size_c * sizeof(int));
            }
        }
        auto end = high_resolution_clock::now();

#pragma omp single
        {
            log_info("Forward cost: %.3lf s, Mem Usage: %s KB",
                     duration_cast<milliseconds>(end - start).count() / 1000.0,
                     FormatWithCommas(getValue()).c_str());
            log_info("Triangle Cnt: %s", FormatWithCommas(triangle_cnt).c_str());
        }
    }
    free(node_off_end);
#endif
}

void SCAN_XP::TriCntRoaring() {
#ifdef ROARING
    vector<Roaring> roarings(g.nodemax);

    auto max_d = 0;
    auto *node_off_end = (uint32_t *) malloc(sizeof(uint32_t) * (g.nodemax + 1));

    long cnt = 0;
#pragma omp parallel num_threads(thread_num_)
    {
#pragma omp for reduction(max: max_d)
        for (auto u = 0u; u < g.nodemax; u++) {
            node_off_end[u + 1] = static_cast<uint32_t>(
                    lower_bound(g.edge_dst + g.node_off[u], g.edge_dst + g.node_off[u + 1], u) - g.edge_dst);
            max_d = max<int>(max_d, node_off_end[u + 1] - g.node_off[u]);
        }

#pragma omp single
        log_info("finish init node_off_end, max d: %d", max_d);

        auto clk_beg = high_resolution_clock::now();
#pragma omp for schedule(dynamic, 100), reduction(+:cnt)
        for (int i = 0; i < g.nodemax; i++) {
            roarings[i] = Roaring(node_off_end[i + 1] - g.node_off[i],
                                  reinterpret_cast<const uint32_t *>(g.node_off[i] + g.edge_dst));
            cnt += roarings[i].runOptimize() ? 1 : 0;
            roarings[i].shrinkToFit();
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("PreProcess-Roaring Cost: %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
            log_info("Run: %s", FormatWithCommas(cnt).c_str());
        }
    }

    uint64_t triangle_cnt = 0;
#pragma omp parallel num_threads(thread_num_)
    {
        auto start = high_resolution_clock::now();
        int *tmp_base = new int[max_d];
        int *tmp_state = new int[max_d];
#pragma omp for schedule(dynamic, 100) reduction(+:triangle_cnt)
        for (auto u = 0u; u < g.nodemax; u++) {
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                triangle_cnt += roaring_bitmap_and_cardinality(&roarings[u].roaring,
                                                               &roarings[v].roaring);
            }
        }
        auto end = high_resolution_clock::now();

#pragma omp single
        {
            log_info("Forward cost: %.3lf s, Mem Usage: %s KB",
                     duration_cast<milliseconds>(end - start).count() / 1000.0,
                     FormatWithCommas(getValue()).c_str());
            log_info("Triangle Cnt: %s", FormatWithCommas(triangle_cnt).c_str());
        }
    }
    free(node_off_end);

#endif
}

void SCAN_XP::TriCntEmptyHeaded() {
#ifdef EMPTY_HEADED
    log_info("Before EmptyHeaded... %s KB", FormatWithCommas(getValue()).c_str());
#ifdef EH_LAYOUT_UINT
    using Layout= uinteger;
#else
    using Layout= hybrid;
#endif
    vector<const Set<Layout> *> eh_sets(g.nodemax);

    auto clk_beg = high_resolution_clock::now();
    size_t max_size = 0;
    size_t type0 = 0;
    size_t type1 = 0;
    size_t max_mem_consumption = 0;

    // pre-processing first for compact-forward.
    auto max_d = 0;
    auto *node_off_end = (uint32_t *) malloc(sizeof(uint32_t) * (g.nodemax + 1));
#pragma omp parallel num_threads(thread_num_)
    {
#pragma omp for reduction(max: max_d)
        for (auto u = 0u; u < g.nodemax; u++) {
            node_off_end[u + 1] = static_cast<uint32_t>(
                    lower_bound(g.edge_dst + g.node_off[u], g.edge_dst + g.node_off[u + 1], u) - g.edge_dst);
            max_d = max<int>(max_d, node_off_end[u + 1] - g.node_off[u]);
        }
#pragma omp single
        log_info("finish init node_off_end, max d: %d", max_d);

#pragma omp for schedule(dynamic, 100), reduction(max: max_size), reduction(+:type0), reduction(+:type1)
        for (int i = 0; i < g.nodemax; i++) {
            eh_sets[i] = NewSet<Layout>(&g, g.node_off[i], node_off_end[i + 1] - g.node_off[i]);
            max_size = max(max_size, eh_sets[i]->number_of_bytes + sizeof(Set<Layout>));
            if (eh_sets[i]->type == 0) {
                type0++;
            } else {
                type1++;
            }
            max_mem_consumption = max(max_mem_consumption, eh_sets[i]->number_of_bytes + sizeof(Set<Layout>));
        }
#pragma omp single
        {
            log_info("type0 (RANGE_BITSET): %s, type1 (UINTEGER): %s, Max Mem: %s KB", FormatWithCommas(type0).c_str(),
                     FormatWithCommas(type1).c_str(), FormatWithCommas(max_mem_consumption).c_str());
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("PreProcess-EmptyHeaded Cost: %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
        }
    }

    uint64_t triangle_cnt = 0;

#pragma omp parallel num_threads(thread_num_)
    {
        auto start = high_resolution_clock::now();

        Set<Layout> *res = (Set<Layout> *) malloc(max_size);

#pragma omp for schedule(dynamic, 100) reduction(+:triangle_cnt)
        for (auto u = 0u; u < g.nodemax; u++) {
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                Set<Layout> *ptr = ops::set_intersect(res, eh_sets[u], eh_sets[v]);
                triangle_cnt += ptr->cardinality;
            }
        }
        auto end = high_resolution_clock::now();

#pragma omp single
        {
            log_info("Forward cost: %.3lf s, Mem Usage: %s KB",
                     duration_cast<milliseconds>(end - start).count() / 1000.0,
                     FormatWithCommas(getValue()).c_str());
            log_info("Triangle Cnt: %s", FormatWithCommas(triangle_cnt).c_str());
        }
    }
    free(node_off_end);

#endif
}

void SCAN_XP::TriCntBitmapOp() {
    auto *node_off_end = (uint32_t *) malloc(sizeof(uint32_t) * (g.nodemax + 1));

    auto max_d = 0;
    uint64_t triangle_cnt = 0;
    uint64_t packed_num = 0;
    using word_type = uint64_t;
    constexpr uint32_t wordinbits = sizeof(word_type) * 8;

    vector<vector<int>> partition_id_lst(g.nodemax);
    vector<vector<word_type>> bitmap_in_partition_lst(g.nodemax);
    uint64_t empty_intersection_num = 0;
    uint64_t empty_index_word_num = 0;

    uint64_t check_word_num = 0;

#pragma omp parallel num_threads(thread_num_)
    {
        // Bit vectors.
        auto bool_arr = BoolArray<word_type>(g.nodemax);

#pragma omp for reduction(max: max_d)
        for (auto u = 0u; u < g.nodemax; u++) {
            node_off_end[u + 1] = static_cast<uint32_t>(
                    lower_bound(g.edge_dst + g.node_off[u], g.edge_dst + g.node_off[u + 1], u) - g.edge_dst);
            max_d = max<int>(max_d, node_off_end[u + 1] - g.node_off[u]);
        }

#pragma omp single
        log_info("finish init node_off_end, max d: %d", max_d);

        auto clk_beg = high_resolution_clock::now();

        // Pre-Process: Indexing Words.
#pragma omp for schedule(dynamic, 100) reduction(+:packed_num)
        for (auto u = 0; u < g.nodemax; u++) {
            auto prev_blk_id = -1;
            auto num_blks = 0;
            auto pack_num_u = 0;
            for (auto off = g.node_off[u]; off < node_off_end[u + 1]; off++) {
                auto v = g.edge_dst[off];
                auto cur_blk_id = v / wordinbits;
                if (cur_blk_id == prev_blk_id) {
                    pack_num_u++;
                } else {
                    prev_blk_id = cur_blk_id;
                    num_blks++;
                }
            }

            prev_blk_id = -1;
            if ((node_off_end[u + 1] - g.node_off[u]) >= 16 && (node_off_end[u + 1] - g.node_off[u]) / num_blks > 2) {
                packed_num++;
                for (auto off = g.node_off[u]; off < node_off_end[u + 1]; off++) {
                    auto v = g.edge_dst[off];
                    auto cur_blk_id = v / wordinbits;
                    if (cur_blk_id == prev_blk_id) {
                        pack_num_u++;
                    } else {
                        prev_blk_id = cur_blk_id;
                        num_blks++;
                        partition_id_lst[u].emplace_back(cur_blk_id);
                        bitmap_in_partition_lst[u].emplace_back(0);
                    }
                    bitmap_in_partition_lst[u].back() |= static_cast<word_type>(1u) << (v % wordinbits);
                }
            }
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("Packed#: %s", FormatWithCommas(packed_num).c_str());
            log_info("PreProcess-BitmapAdvanced Cost: %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
        }

        auto start = high_resolution_clock::now();

#pragma omp for schedule(dynamic, 100) reduction(+:triangle_cnt) reduction(+:check_word_num) reduction(+:empty_intersection_num) \
reduction(+:empty_index_word_num)
        for (auto u = 0u; u < g.nodemax; u++) {
            // Set.
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                bool_arr.set(v);
            }
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                if (!partition_id_lst[v].empty()) {
                    for (auto wi = 0; wi < partition_id_lst[v].size(); wi++) {
                        auto index_word = bool_arr.getWord(partition_id_lst[v][wi]);
                        auto res = index_word & bitmap_in_partition_lst[v][wi];
                        if (index_word == 0) {
                            empty_index_word_num++;
                        }
                        if (res == 0) {
                            empty_intersection_num++;
                        }
                        triangle_cnt += popcnt(&res, sizeof(word_type));
                    }
                    check_word_num += partition_id_lst[v].size();

                } else if (node_off_end[v + 1] - g.node_off[v] > 0) {
                    auto prev_blk_id = g.edge_dst[g.node_off[v]] / wordinbits;
                    word_type bitmap_pack = 0;
                    for (auto off = g.node_off[v]; off < node_off_end[v + 1]; off++) {
                        auto w = g.edge_dst[off];
                        // Online pack.
                        auto cur_blk_id = w / wordinbits;
                        if (cur_blk_id != prev_blk_id) {
                            // Execute prev.
                            auto index_word = bool_arr.getWord(prev_blk_id);
                            auto res = index_word & bitmap_pack;
                            if (index_word == 0) {
                                empty_index_word_num++;
                            }
                            if (res == 0) {
                                empty_intersection_num++;
                            }
                            triangle_cnt += popcnt(&res, sizeof(word_type));

                            check_word_num++;
                            bitmap_pack = 0;
                            prev_blk_id = cur_blk_id;
                        }
                        bitmap_pack |= static_cast<word_type >(1u) << (w % wordinbits);
                    }
//                    // Execute last.
                    auto index_word = bool_arr.getWord(prev_blk_id);
                    auto res = index_word & bitmap_pack;
                    if (index_word == 0) {
                        empty_index_word_num++;
                    }
                    if (res == 0) {
                        empty_intersection_num++;
                    }
                    check_word_num++;
                    triangle_cnt += popcnt(&res, sizeof(word_type));
                }
            }
            // Clear.
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                bool_arr.setWord(v / wordinbits, 0);
            }
        }
        auto end = high_resolution_clock::now();

#pragma omp single
        {
            log_info("Forward cost: %.3lf s, Mem Usage: %s KB",
                     duration_cast<milliseconds>(end - start).count() / 1000.0,
                     FormatWithCommas(getValue()).c_str());
            log_info("Triangle Cnt: %s", FormatWithCommas(triangle_cnt).c_str());
            log_debug("Check Word Num: %s (Bits: %s), TC: %s, Valid/Check: %.6lf",
                      FormatWithCommas(check_word_num).c_str(),
                      FormatWithCommas(check_word_num * wordinbits).c_str(),
                      FormatWithCommas(triangle_cnt).c_str(),
                      static_cast<double>(triangle_cnt) / (check_word_num * wordinbits));
            log_debug("Empty Num: %s (Empty Idx: %s) (EmptyIdx/All, Empty/All: %.6lf, %.6lf)",
                      FormatWithCommas(empty_intersection_num).c_str(),
                      FormatWithCommas(empty_index_word_num).c_str(),
                      static_cast<double>(empty_index_word_num) / check_word_num,
                      static_cast<double>(empty_intersection_num) / check_word_num);
            log_debug("Valid/Check-RF: %.6lf (Hits: %.6lf)",
                      static_cast<double>(triangle_cnt) / ((check_word_num - empty_index_word_num) * wordinbits),
                      static_cast<double>(triangle_cnt) / ((check_word_num - empty_index_word_num)));
        }
    }
    free(node_off_end);
}

void SCAN_XP::TriCntBitmapAdv() {
    auto *node_off_end = (uint32_t *) malloc(sizeof(uint32_t) * (g.nodemax + 1));

    auto max_d = 0;
    uint64_t triangle_cnt = 0;
    uint64_t packed_num = 0;
    using word_type = uint64_t;
    constexpr uint32_t wordinbits = sizeof(word_type) * 8;

    vector<vector<int>> partition_id_lst(g.nodemax);
    vector<vector<word_type>> bitmap_in_partition_lst(g.nodemax);

#pragma omp parallel num_threads(thread_num_)
    {
        // Bit vectors.
        auto bool_arr = BoolArray<word_type>(g.nodemax);

#pragma omp for reduction(max: max_d)
        for (auto u = 0u; u < g.nodemax; u++) {
            node_off_end[u + 1] = static_cast<uint32_t>(
                    lower_bound(g.edge_dst + g.node_off[u], g.edge_dst + g.node_off[u + 1], u) - g.edge_dst);
            max_d = max<int>(max_d, node_off_end[u + 1] - g.node_off[u]);
        }

#pragma omp single
        log_info("finish init node_off_end, max d: %d", max_d);

        auto clk_beg = high_resolution_clock::now();

        // Pre-Process: Indexing Words.
#pragma omp for schedule(dynamic, 100) reduction(+:packed_num)
        for (auto u = 0; u < g.nodemax; u++) {
            auto prev_blk_id = -1;
            auto num_blks = 0;
            auto pack_num_u = 0;
            for (auto off = g.node_off[u]; off < node_off_end[u + 1]; off++) {
                auto v = g.edge_dst[off];
                auto cur_blk_id = v / wordinbits;
                if (cur_blk_id == prev_blk_id) {
                    pack_num_u++;
                } else {
                    prev_blk_id = cur_blk_id;
                    num_blks++;
                }
            }

            prev_blk_id = -1;
            if ((node_off_end[u + 1] - g.node_off[u]) >= 16 && (node_off_end[u + 1] - g.node_off[u]) / num_blks > 2) {
                packed_num++;
                for (auto off = g.node_off[u]; off < node_off_end[u + 1]; off++) {
                    auto v = g.edge_dst[off];
                    auto cur_blk_id = v / wordinbits;
                    if (cur_blk_id == prev_blk_id) {
                        pack_num_u++;
                    } else {
                        prev_blk_id = cur_blk_id;
                        num_blks++;
                        partition_id_lst[u].emplace_back(cur_blk_id);
                        bitmap_in_partition_lst[u].emplace_back(0);
                    }
                    bitmap_in_partition_lst[u].back() |= static_cast<word_type>(1u) << (v % wordinbits);
                }
            }
        }

#pragma omp single
        {
            auto clk_end = high_resolution_clock::now();
            auto time = duration_cast<nanoseconds>(clk_end - clk_beg).count();
            log_info("Packed#: %s", FormatWithCommas(packed_num).c_str());
            log_info("PreProcess-BitmapAdvanced Cost: %.9lf s, %s KB", time / pow(10, 9),
                     FormatWithCommas(getValue()).c_str());
        }

        auto start = high_resolution_clock::now();

#pragma omp for schedule(dynamic, 100) reduction(+:triangle_cnt)
        for (auto u = 0u; u < g.nodemax; u++) {
            // Set.
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                bool_arr.set(v);
            }
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                if (!partition_id_lst[v].empty()) {
                    for (auto wi = 0; wi < partition_id_lst[v].size(); wi++) {
                        auto res = bool_arr.getWord(partition_id_lst[v][wi]) & bitmap_in_partition_lst[v][wi];
                        triangle_cnt += popcnt(&res, sizeof(word_type));
                    }
                } else {
                    for (auto off = g.node_off[v]; off < node_off_end[v + 1]; off++) {
                        auto w = g.edge_dst[off];
                        if (bool_arr.get(w))
                            triangle_cnt++;
                    }
                }
            }
            // Clear.
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                bool_arr.setWord(v / wordinbits, 0);
            }
        }
        auto end = high_resolution_clock::now();

#pragma omp single
        {
            log_info("Forward cost: %.3lf s, Mem Usage: %s KB",
                     duration_cast<milliseconds>(end - start).count() / 1000.0,
                     FormatWithCommas(getValue()).c_str());
            log_info("Triangle Cnt: %s", FormatWithCommas(triangle_cnt).c_str());
        }
    }
    free(node_off_end);
}

void SCAN_XP::TriCntBitmap() {
    auto *node_off_end = (uint32_t *) malloc(sizeof(uint32_t) * (g.nodemax + 1));

    auto max_d = 0;
    uint64_t triangle_cnt = 0;

#pragma omp parallel num_threads(thread_num_)
    {
        // Bit vectors.
        auto bits_vec = vector<bool>(g.nodemax, false);
#ifdef BIT_VEC_INDEX
        auto index_bits_vec = vector<bool>((g.nodemax + INDEX_RANGE - 1) / INDEX_RANGE, false);
#endif

#pragma omp for reduction(max: max_d)
        for (auto u = 0u; u < g.nodemax; u++) {
            node_off_end[u + 1] = static_cast<uint32_t>(
                    lower_bound(g.edge_dst + g.node_off[u], g.edge_dst + g.node_off[u + 1], u) - g.edge_dst);
            max_d = max<int>(max_d, node_off_end[u + 1] - g.node_off[u]);
        }

#pragma omp single
        log_info("finish init node_off_end, max d: %d", max_d);

        auto start = high_resolution_clock::now();

#pragma omp for schedule(dynamic, 100) reduction(+:triangle_cnt)
        for (auto u = 0u; u < g.nodemax; u++) {
            // Set.
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                bits_vec[v] = true;
#ifdef BIT_VEC_INDEX
                index_bits_vec[v >> INDEX_BIT_SCALE_LOG] = true;
#endif
            }

            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
#ifndef BIT_VEC_INDEX
                triangle_cnt += ComputeCNHashBitVec(&g, g.node_off[v], node_off_end[v + 1], bits_vec);
#else
                triangle_cnt += ComputeCNHashBitVec2D(&g, g.node_off[v], node_off_end[v + 1], bits_vec, index_bits_vec);
#endif
            }

            // Clear.
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                bits_vec[v] = false;
            }
#ifdef BIT_VEC_INDEX
            index_bits_vec.assign(index_bits_vec.size(), false);
#endif
        }
        auto end = high_resolution_clock::now();

#pragma omp single
        {
            log_info("Forward cost: %.3lf s, Mem Usage: %s KB",
                     duration_cast<milliseconds>(end - start).count() / 1000.0,
                     FormatWithCommas(getValue()).c_str());
            log_info("Triangle Cnt: %s", FormatWithCommas(triangle_cnt).c_str());
        }
    }
    free(node_off_end);
}

void SCAN_XP::TriCntHash() {
    auto *node_off_end = (uint32_t *) malloc(sizeof(uint32_t) * (g.nodemax + 1));

    auto max_d = 0;
    uint64_t triangle_cnt = 0;

#pragma omp parallel num_threads(thread_num_)
    {

#if defined(HASH_SPP)
        spp::sparse_hash_set<int> hash_table;
#else
        unordered_set<int> hash_table;
#endif

#pragma omp for reduction(max: max_d)
        for (auto u = 0u; u < g.nodemax; u++) {
            node_off_end[u + 1] = static_cast<uint32_t>(
                    lower_bound(g.edge_dst + g.node_off[u], g.edge_dst + g.node_off[u + 1], u) - g.edge_dst);
            max_d = max<int>(max_d, node_off_end[u + 1] - g.node_off[u]);
        }

#pragma omp single
        log_info("finish init node_off_end, max d: %d", max_d);

        auto start = high_resolution_clock::now();
#pragma omp for schedule(dynamic, 100) reduction(+:triangle_cnt)
        for (auto u = 0u; u < g.nodemax; u++) {
            // Set.
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                hash_table.emplace(v);
            }

            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];
                triangle_cnt += ComputeCNHash(&g, g.node_off[v], node_off_end[v + 1], hash_table);
            }

            // Clear.
            hash_table.clear();
        }
        auto end = high_resolution_clock::now();

#pragma omp single
        {
            log_info("Forward cost: %.3lf s, Mem Usage: %s KB",
                     duration_cast<milliseconds>(end - start).count() / 1000.0,
                     FormatWithCommas(getValue()).c_str());
            log_info("Triangle Cnt: %s", FormatWithCommas(triangle_cnt).c_str());
        }
    }
    free(node_off_end);
}

void SCAN_XP::TriCntSortedArray() {
    auto *node_off_end = (uint32_t *) malloc(sizeof(uint32_t) * (g.nodemax + 1));

    auto max_d = 0;
    uint64_t triangle_cnt = 0;

    int64_t skew = 0;
    int64_t total = 0;

#pragma omp parallel num_threads(thread_num_)
    {
        auto *my_ptr = new uint32_t[1024 * 1024 * 2];
        static thread_local int *my_ptr_han = new int[1024 * 1024 * 2];

#pragma omp for reduction(max: max_d)
        for (auto u = 0u; u < g.nodemax; u++) {
            node_off_end[u + 1] = static_cast<uint32_t>(
                    lower_bound(g.edge_dst + g.node_off[u], g.edge_dst + g.node_off[u + 1], u) - g.edge_dst);
            max_d = max<int>(max_d, node_off_end[u + 1] - g.node_off[u]);
        }

#pragma omp single
        log_info("finish init node_off_end, max d: %d", max_d);

        auto start = high_resolution_clock::now();
#pragma omp for schedule(dynamic, 100) reduction(+:triangle_cnt) reduction(+:total) reduction(+:skew)
        for (auto u = 0u; u < g.nodemax; u++) {
            for (auto edge_idx = g.node_off[u]; edge_idx < node_off_end[u + 1]; edge_idx++) {
                auto v = g.edge_dst[edge_idx];

#if defined(NAIVE)
                print_str("NAIVE");
                auto size = ComputeCNNaive(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                          node_off_end[v + 1]);
#elif defined(NAIVE_MERGE)
                print_str("NAIVE_MERGE");
                auto size = ComputeCNNaiveStdMerge(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                          node_off_end[v + 1]);
#elif defined(NAIVE_GALLOPING)
                print_str("NAIVE_GALLOPING");
                auto size = ComputeCNGallopingSingleDir(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                          node_off_end[v + 1]);
#elif defined(NAIVE_HYBRID)
                print_str("NAIVE_HYBRID");
                auto size = ComputeCNGallopingSingleDir(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                          node_off_end[v + 1]);
#elif defined(TETZANK_AVX)
                print_str("TETZANK_AVX");
                auto size = intersect_vector_avx_count(
                        reinterpret_cast<const uint32_t *>(g.node_off[u] + g.edge_dst),
                        node_off_end[u + 1] - g.node_off[u],
                        reinterpret_cast<const uint32_t *>(g.node_off[v] + g.edge_dst),
                        node_off_end[v + 1] - g.node_off[v]);
#elif defined(TETZANK_AVX2)
                print_str("TETZANK_AVX2");
                auto size = intersect_vector_avx2_count(
                        reinterpret_cast<const uint32_t *>(g.node_off[u] + g.edge_dst),
                        node_off_end[u + 1] - g.node_off[u],
                        reinterpret_cast<const uint32_t *>(g.node_off[v] + g.edge_dst),
                        node_off_end[v + 1] - g.node_off[v]);
#elif defined(TETZANK_AVX2_ASM)
                print_str("TETZANK_AVX2_ASM");
                auto size = intersect_vector_avx2_asm_count(
                        reinterpret_cast<const uint32_t *>(g.node_off[u] + g.edge_dst),
                        node_off_end[u + 1] - g.node_off[u],
                        reinterpret_cast<const uint32_t *>(g.node_off[v] + g.edge_dst),
                        node_off_end[v + 1] - g.node_off[v]);
#elif defined(HAN_QFILTER)
                print_str("HAN_QFILTER");
                auto size = intersect_qfilter_uint_b4_v2(
                        g.node_off[u] + g.edge_dst,
                        node_off_end[u + 1] - g.node_off[u],
                        g.node_off[v] + g.edge_dst,
                        node_off_end[v + 1] - g.node_off[v],
                        my_ptr_han);
#elif defined(HAN_BMISS)
                print_str("HAN_BMISS");
                auto size = intersect_bmiss_uint_b4(
                        g.node_off[u] + g.edge_dst,
                        node_off_end[u + 1] - g.node_off[u],
                        g.node_off[v] + g.edge_dst,
                        node_off_end[v + 1] - g.node_off[v],
                        my_ptr_han);
#elif defined(HAN_BMISS_STTNI)
                print_str("HAN_BMISS_STTNI");
                auto size = intersect_bmiss_uint_sttni_b8(
                        g.node_off[u] + g.edge_dst,
                        node_off_end[u + 1] - g.node_off[u],
                        g.node_off[v] + g.edge_dst,
                        node_off_end[v + 1] - g.node_off[v],
                        my_ptr_han);
#elif defined(HAN_HIER)
                print_str("HAN_HIER");
                auto size = intersect_hierainter_uint_sttni(
                        g.node_off[u] + g.edge_dst,
                        node_off_end[u + 1] - g.node_off[u],
                        g.node_off[v] + g.edge_dst,
                        node_off_end[v + 1] - g.node_off[v],
                        my_ptr_han);
#elif defined(SSE_MERGE)
                print_str("SSE_MERGE");
                auto size = ComputeCNSSE4(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                          node_off_end[v + 1]);
#elif defined(SSE_HYBRID)
                print_str("SSE_HYBRID");
                auto size = ComputeCNSSEHybrid(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                          node_off_end[v + 1]);
#elif defined(SSE_PIVOT)
                print_str("SSE_PIVOT");
                auto size = ComputeCNPivotSSE4(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                          node_off_end[v + 1]);
#elif defined(AVX2)
                print_str("AVX2");
                auto size = ComputeCNAVX2(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                          node_off_end[v + 1]);
#elif defined(AVX2_PIVOT)
                print_str("AVX2_PIVOT");
                auto size = ComputeCNPivotAVX2(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                                            node_off_end[v + 1]);
#elif defined(AVX2_GALLOPING_SINGLE)
                print_str("AVX2_GALLOPING_SINGLE");
                auto size = ComputeCNGallopingSingleDirAVX2(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                                            node_off_end[v + 1]);
#elif defined(AVX2_HYBRID)
                print_str("AVX2_HYBRID");
                auto size = ComputeCNHybridAVX2(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                                            node_off_end[v + 1]);
#elif defined(LEMIRE_SSE)
                print_str("LEMIRE_SSE");
                auto size =  SIMDCompressionLib::SIMDintersection(
                        reinterpret_cast<const uint32_t *>(g.node_off[u] + g.edge_dst),
                        node_off_end[u + 1] - g.node_off[u],
                        reinterpret_cast<const uint32_t *>(g.node_off[v] + g.edge_dst),
                        node_off_end[v + 1] - g.node_off[v],
                        my_ptr);
#elif defined(LEMIRE_AVX2)
                print_str("LEMIRE_AVX2");
                auto size =  SIMDCompressionLib::SIMDintersection_avx2(
                        reinterpret_cast<const uint32_t *>(g.node_off[u] + g.edge_dst),
                        node_off_end[u + 1] - g.node_off[u],
                        reinterpret_cast<const uint32_t *>(g.node_off[v] + g.edge_dst),
                        node_off_end[v + 1] - g.node_off[v],
                        my_ptr);
#elif defined(LEMIRE_HIGHLY_SCALABLE)
                print_str("LEMIRE_HIGHLY_SCALABLE");
                auto size =  SIMDCompressionLib::lemire_highlyscalable_intersect_SIMD(
                        reinterpret_cast<const uint32_t *>(g.node_off[u] + g.edge_dst),
                        node_off_end[u + 1] - g.node_off[u],
                        reinterpret_cast<const uint32_t *>(g.node_off[v] + g.edge_dst),
                        node_off_end[v + 1] - g.node_off[v],
                        my_ptr);
#else
                print_str("NAIVE");
                auto size = ComputeCNNaive(&g, g.node_off[u], node_off_end[u + 1], g.node_off[v],
                                          node_off_end[v + 1]);
#endif

#define SKEW_THRESHOLD (32)
                total++;
                auto d1 = node_off_end[u + 1] - g.node_off[u] + 1;
                auto d2 = node_off_end[v + 1] - g.node_off[v] + 1;
                if (d1 > SKEW_THRESHOLD * d2 || d2 > SKEW_THRESHOLD * d1) {
                    skew++;
                }
                triangle_cnt += size;
            }
        }
        auto end = high_resolution_clock::now();

#pragma omp single
        {
            log_info("Forward cost: %.3lf s, Mem Usage: %s KB",
                     duration_cast<milliseconds>(end - start).count() / 1000.0,
                     FormatWithCommas(getValue()).c_str());
            log_info("Triangle Cnt: %s", FormatWithCommas(triangle_cnt).c_str());
            log_info("Skew: %s, Total: %s", FormatWithCommas(skew).c_str(), FormatWithCommas(total).c_str());
        }
    }
    free(node_off_end);
}
