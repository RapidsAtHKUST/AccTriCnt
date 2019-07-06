#include "scan_xp.h"

#include <cmath>

#include <chrono>
#include <thread>

#include "util/stat.h"
#include "util/log.h"
#include "util/util.h"
#include "util/pretty_print.h"

using namespace std::chrono;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

SCAN_XP::SCAN_XP(int thread_num, int min_u, double epsilon, char *dir) :
		thread_num_(thread_num), min_u_(min_u), epsilon_(epsilon), g(Graph(dir)), uf_ptr(nullptr) {
	log_info("thread num: %d", thread_num_);log_info("graph, n: %s, m: %s", FormatWithCommas(g.nodemax).c_str(), FormatWithCommas(g.edgemax).c_str());

	uf_ptr = new UnionFind(g.nodemax);

	// pre-processing
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic, 100000)
	for (auto i = 0u; i < g.nodemax; i++) {
		g.core_count[i] = 0;
		g.label[i] = UNCLASSIFIED;
		for (auto n = g.node_off[i]; n < g.node_off[i + 1]; n++) {
			g.common_node_num[n] = 2;
		}
	}
}

SCAN_XP::~SCAN_XP() {
	delete uf_ptr;
}

uint32_t SCAN_XP::BinarySearch(int *array, uint32_t offset_beg, uint32_t offset_end, int val) {
	auto mid = static_cast<uint32_t>((static_cast<unsigned long>(offset_beg) + offset_end) / 2);
	if (array[mid] == val) {
		return mid;
	}
	return val < array[mid] ? BinarySearch(array, offset_beg, mid, val) : BinarySearch(array, mid + 1, offset_end, val);
}

int SCAN_XP::FindSrc(Graph *g, int u, uint32_t edge_idx) {
	if (edge_idx >= g->node_off[u + 1]) {
		// update last_u, preferring galloping instead of binary search because not large range here
		u = GallopingSearch(g->node_off, static_cast<uint32_t>(u) + 1, g->nodemax + 1, edge_idx);
		// 1) first > , 2) has neighbor
		if (g->node_off[u] > edge_idx) {
			while (g->degree[u - 1] == 1) {
				u--;
			}
			u--;
		} else {
			// g->node_off[u] == i
			while (g->degree[u] == 1) {
				u++;
			}
		}
	}
	return u;
}

void SCAN_XP::ClusterCore() {
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic, 2000)
	for (auto i = 0u; i < g.nodemax; i++) {
		if (g.label[i] == CORE) {
			for (auto edge_idx = g.node_off[i]; edge_idx < g.node_off[i + 1]; edge_idx++) {
				// yche: fix bug, only union when g.edge_dst[edge_idx] is a core vertex
				if (g.label[g.edge_dst[edge_idx]] == CORE && g.similarity[edge_idx]) {
					uf_ptr->UnionThreadSafe(i, g.edge_dst[edge_idx]);
				}
			}
		}
	}
}

bool SCAN_XP::CheckHub(Graph *g, UnionFind *uf, int a) {
	set<int> c;

	for (auto i = g->node_off[a]; i < g->node_off[a + 1]; i++) {
		if (g->label[g->edge_dst[i]] != CORE)
			continue;
		c.insert((*uf).FindRoot(g->edge_dst[i]));
	}
	return c.size() >= 2;
}

void SCAN_XP::LabelNonCore() {
	int core_num = 0u;
#pragma omp parallel for num_threads(thread_num_) schedule(dynamic, 1000), reduction(+:core_num)
	for (auto i = 0u; i < g.nodemax; i++) {
		if (g.label[i] == CORE) {
			core_num++;
			continue;
		}

		if (CheckHub(&g, uf_ptr, i)) {
			g.label[i] = HUB;
		}
	}log_info("Core: %s", FormatWithCommas(core_num).c_str());
}

void SCAN_XP::PostProcess() {
	int cluster_num;
	int hub_num = 0u;
	int out_num = 0u;

	set<int> c;
	for (auto i = 0u; i < g.nodemax; i++) {
		if (g.label[i] == CORE) {
			c.emplace(uf_ptr->FindRoot(i));
		}
	}
	cluster_num = static_cast<int>(c.size());

#pragma omp parallel for num_threads(thread_num_) reduction(+:hub_num, out_num)
	for (auto i = 0u; i < g.nodemax; i++) {
		if (g.label[i] == HUB) {
			hub_num++;
		} else if (g.label[i] == UNCLASSIFIED) {
			out_num++;
		}
	}log_info("Cluster: %s, Hub: %s, Outlier: %s", FormatWithCommas(cluster_num).c_str(),
			FormatWithCommas(hub_num).c_str(), FormatWithCommas(out_num).c_str());
}

void SCAN_XP::MarkClusterMinEleAsId(UnionFind *union_find_ptr) {
	auto start = high_resolution_clock::now();
	g.cluster_dict = vector<int>(g.nodemax);
	std::fill(g.cluster_dict.begin(), g.cluster_dict.end(), g.nodemax);

#pragma omp parallel for num_threads(thread_num_)
	for (auto i = 0u; i < g.nodemax; i++) {
		if (g.label[i] == CORE) {
			int x = union_find_ptr->FindRoot(i);
			int cluster_min_ele;
			do {
				cluster_min_ele = g.cluster_dict[x];
				if (i >= g.cluster_dict[x]) {
					break;
				}
			} while (!__sync_bool_compare_and_swap(&(g.cluster_dict[x]), cluster_min_ele, i));
		}
	}
	auto end = high_resolution_clock::now();
	log_info("Step4 - cluster id initialization cost: %.3lf s, Mem Usage: %s KB",
			duration_cast<milliseconds>(end - start).count() / 1000.0, FormatWithCommas(getValue()).c_str());
}

void SCAN_XP::PrepareResultOutput() {
// prepare output
	MarkClusterMinEleAsId(uf_ptr);

	auto start = high_resolution_clock::now();
#pragma omp parallel num_threads(thread_num_)
	{
		vector<pair<int, int>> local_non_core_cluster;
#pragma omp for nowait
		for (auto i = 0u; i < g.nodemax; i++) {
			if (g.label[i] == CORE) {
				for (auto j = g.node_off[i]; j < g.node_off[i + 1]; j++) {
					auto v = g.edge_dst[j];
					if (g.label[v] != CORE && g.similarity[j]) {
						local_non_core_cluster.emplace_back(g.cluster_dict[uf_ptr->FindRoot(i)], v);
					}
				}
			}
		}
#pragma omp critical
		{
			for (auto ele : local_non_core_cluster) {
				g.noncore_cluster.emplace_back(ele);
			}
		};

	};
	auto end = high_resolution_clock::now();
	log_info("Step4 - prepare results: %.3lf s, Mem Usage: %s KB",
			duration_cast<milliseconds>(end - start).count() / 1000.0, FormatWithCommas(getValue()).c_str());

	auto epsilon_str = to_string(epsilon_);
	epsilon_str.erase(epsilon_str.find_last_not_of("0u") + 1);

	start = high_resolution_clock::now();
	g.Output(epsilon_str.c_str(), to_string(min_u_).c_str(), uf_ptr);
	end = high_resolution_clock::now();
	log_info("Step4 - output to the disk cost: %.3lf s, Mem Usage: %s KB",
			duration_cast<milliseconds>(end - start).count() / 1000.0, FormatWithCommas(getValue()).c_str());
}

bool CheckSubstring(std::string firstString, std::string secondString) {
	if (secondString.size() > firstString.size())
		return false;

	for (int i = 0; i < firstString.size(); i++) {
		int j = 0;
		if (firstString[i] == secondString[j]) {
			while (firstString[i] == secondString[j] && j < secondString.size()) {
				j++;
				i++;
			}

			if (j == secondString.size())
				return true;
		}
	}
	return false;
}

void PostComputeCoreChecking(SCAN_XP* scanxp, Graph* g, int min_u_, double epsilon_) {
	{
		auto start = high_resolution_clock::now();
#pragma omp parallel for num_threads(scanxp->thread_num_) schedule(dynamic, 60000)
		for (auto i = 0u; i < g->edgemax; i++) {
			// remove edge_src optimization, assuming task scheduling in FIFO-queue-mode
			static thread_local auto u = 0;
			u = scanxp->FindSrc(g, u, i);
			auto v = g->edge_dst[i];
			if (u > v) {
				uint32_t offset = g->common_node_num[i];
				g->common_node_num[i] = g->common_node_num[offset];
			}
		}
		auto end = high_resolution_clock::now();
		log_info("bin-search cost : %.3lf s, Mem Usage: %s KB",
				duration_cast<milliseconds>(end - start).count() / 1000.0, FormatWithCommas(getValue()).c_str());
	}

	// 5th: compute similarities and check core status
	{
		auto start = high_resolution_clock::now();
#pragma omp parallel for num_threads(scanxp->thread_num_)
		for (auto i = 0u; i < g->edgemax; i++) {
			static thread_local auto u = 0;
			u = scanxp->FindSrc(g, u, i);

			long double du = g->node_off[u + 1] - g->node_off[u] + 1;
			long double dv = g->node_off[g->edge_dst[i] + 1] - g->node_off[g->edge_dst[i]] + 1;
			auto sim_value = static_cast<double>((long double) g->common_node_num[i] / sqrt(du * dv));
			if (sim_value >= epsilon_) {
				//#pragma omp atomic		// comment out assuming no overlap because of large range
				g->core_count[u]++;
				g->similarity[i] = true;
			} else {
				g->similarity[i] = false;
			}
		}
#pragma omp parallel for num_threads(scanxp->thread_num_)
		for (auto i = 0u; i < g->nodemax; i++) {
			if (g->core_count[i] >= min_u_) {
				g->label[i] = CORE;
			};
		}
		auto end = high_resolution_clock::now();
		log_info("core-checking sim-core-comp cost : %.3lf s, Mem Usage: %s KB",
				duration_cast<milliseconds>(end - start).count() / 1000.0, FormatWithCommas(getValue()).c_str());
	}
}

void Verify(SCAN_XP* scanxp, Graph* g) {
	double s1 = omp_get_wtime();

	log_info("Verify with BitVec");
	//     iterate through vertices
#pragma omp parallel for num_threads(scanxp->thread_num_) schedule(dynamic, 6000)
	for (auto i = 0u; i < g->edgemax; i++) {
		// remove edge_src optimization, assuming task scheduling in FIFO-queue-mode
		static thread_local auto u = 0;
		u = scanxp->FindSrc(g, u, i);
		auto v = g->edge_dst[i];
#if defined(BIT_VEC)
		static thread_local auto last_u = -1;
		static thread_local auto bits_vec = vector<bool>(g->nodemax, false);

		if (last_u != u) {
			// clear previous
			if (last_u != -1)
			{
				for (auto offset = g->node_off[last_u]; offset < g->node_off[last_u + 1]; offset++) {
					bits_vec[g->edge_dst[offset]] = false;
				}
			}
			// set new ones
			for (auto offset = g->node_off[u]; offset < g->node_off[u + 1]; offset++) {
				bits_vec[g->edge_dst[offset]] = true;
			}
			last_u = u;
		}
#endif
		if (u < v) {
#ifdef BIT_VEC
			if(g->common_node_num[i]!=2+ComputeCNHashBitVec(g, u, v, bits_vec))
			{
#pragma omp critical
				{
					log_info("Verify Fail, Fix Bug Now!");
					log_info("%d, %d, cur: %d, exp: %d", u, v, g->common_node_num[i],
							2 + ComputeCNHashBitVec(g, u, v, bits_vec));
					vector<int> u_nei;
					for (auto i = g->node_off[u]; i < g->node_off[u + 1]; i++) {
						u_nei.push_back(g->edge_dst[i]);
					}
					cout << u_nei << endl;
					u_nei.clear();
					for (auto i = g->node_off[v]; i < g->node_off[v + 1]; i++) {
						u_nei.push_back(g->edge_dst[i]);
					}

					cout << u_nei << endl;
					exit(-1);
				}
			}
#else
			log_info("err");
			exit(-1);
#endif
		}
	}

	double e1 = omp_get_wtime();
	log_info("Verification Time Cost: %.3lf s, Mem Usage: %s KB", e1 - s1, FormatWithCommas(getValue()).c_str());log_info("Verify Pass, Correct!");
}

void SCAN_XP::Execute() {
//step1 CheckCore
	double s1 = omp_get_wtime();
	CheckCore(&g);
	double e1 = omp_get_wtime();
	log_info("Step1 - CheckCore: %.3lf s, Mem Usage: %s KB", e1 - s1, FormatWithCommas(getValue()).c_str());

//#ifdef VERIFY_WITH_CPU
	Verify(this, &g);
//#endif

//step2 ClusterCore
	double s2 = omp_get_wtime();
	ClusterCore();
	double e2 = omp_get_wtime();
	log_info("Step2 - ClusterCore: %.3lf s, Mem Usage: %s KB", e2 - s2, FormatWithCommas(getValue()).c_str());

//step3 LabelNonCore
	double s3 = omp_get_wtime();
	LabelNonCore();
	double e3 = omp_get_wtime();
	log_info("Step3 - LabelNonCore: %.3lf s, Mem Usage: %s KB", e3 - s3, FormatWithCommas(getValue()).c_str());

// post-processing, prepare result and output
	PostProcess();
	PrepareResultOutput();
}
