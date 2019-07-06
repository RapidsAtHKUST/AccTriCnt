#include "graph.h"

#include <sys/mman.h>
#include <unistd.h>

#ifdef MULTI_GPU
#include "../cuda-utils/cuda_util.cuh"
#include <omp.h>
#endif  

#include <cstring>

#include <chrono>
#include <algorithm>
#include <fcntl.h>

#ifdef TBB

#include <tbb/parallel_sort.h>
//#include <tbb/parallel_scan.h>

#endif

#include "log.h"
#include "util.h"

using namespace chrono;

uint32_t ** d_bitmaps_arr;
uint32_t ** d_vertex_count_arr;
uint32_t ** d_bitmap_states_arr;

Graph::Graph(char *dir_cstr) {
	dir = string(dir_cstr);

	ReadDegree();
	ReadAdjacencyList();
	CheckInputGraph();

	// vertex property
	label = new int[nodemax];
	core_count = new int[nodemax];

	// edge property
#ifndef UNIFIED_MEM
	common_node_num = new int[edgemax];
#else
	cudaMallocManaged(&common_node_num, edgemax * sizeof(int));
#endif
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

/*for bitmap-based set intersection*/
#define BITMAP_SCALE_LOG (9)
#define BITMAP_SCALE (1<<BITMAP_SCALE_LOG)  /*#bits in the first-level bitmap indexed by 1 bit in the second-level bitmap*/

void Graph::ReadAdjacencyList() {
	auto start = high_resolution_clock::now();
	ifstream adj_file(dir + string("/b_adj.bin"), ios::binary);

	// csr representation
#ifndef UNIFIED_MEM
	node_off = new uint32_t[nodemax + 1];
#else
	cudaMallocManaged(&node_off, (nodemax+1) * sizeof(uint32_t));
#endif

#if defined(VARY_BLOCK_SIZE) && defined(MULTI_GPU) && defined(USE_BITMAP_KERNEL)
	extern int num_of_warps_global;
	uint32_t block_size = 32 * num_of_warps_global;
	log_info("block size: %d", block_size);
#elif defined(MULTI_GPU) && defined(USE_BITMAP_KERNEL)
	uint32_t block_size = 128;
#endif


#ifdef MULTI_GPU
	log_info("multi-gpu advise node_off as read-mostly");
	auto num_of_gpus = 1;
	cudaGetDeviceCount(&num_of_gpus);
	for(auto i = 0; i< num_of_gpus; i++) {
		cudaMemAdvise(node_off, (nodemax+1) * sizeof(uint32_t), cudaMemAdviseSetReadMostly ,i);
		cudaMemPrefetchAsync(node_off, (nodemax+1) * sizeof(uint32_t), i, NULL);
	}
#endif

#if defined(MULTI_GPU) &&  defined(USE_BITMAP_KERNEL)
	d_bitmaps_arr = (uint32_t**) malloc(num_of_gpus * sizeof(uint32_t*));
	d_vertex_count_arr = (uint32_t**) malloc(num_of_gpus * sizeof(uint32_t*));
	d_bitmap_states_arr = (uint32_t**) malloc(num_of_gpus * sizeof(uint32_t*));

#pragma omp parallel num_threads(num_of_gpus)
	{
		auto gpu_id = omp_get_thread_num();
		cudaSetDevice(gpu_id);
		//		uint32_t *d_bitmaps, *d_vertex_count;
		/*get the maximal number of threads in an SM*/
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, gpu_id); /*currently 0th device*/
		uint32_t max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
		uint32_t num_SMs = prop.multiProcessorCount;

		uint32_t conc_blocks_per_SM = std::min<int>(max_threads_per_SM / block_size, 16); /*assume regs are not limited*/
		/*initialize the bitmaps*/
		const uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/
		const uint32_t val_size_bitmap = (nodemax + elem_bits - 1) / elem_bits;
		const uint32_t val_size_bitmap_indexes = (val_size_bitmap + BITMAP_SCALE - 1) / BITMAP_SCALE;

		log_info("n_C: %d", max_threads_per_SM);
		log_info("num_SMs: %d", num_SMs);
		log_info("val bitmap size: %d", val_size_bitmap);
		log_info("bitmap bytes: %s Bytes", FormatWithCommas(
						static_cast<uint64_t>(conc_blocks_per_SM) * num_SMs * val_size_bitmap * sizeof(uint32_t)).c_str());
		log_info("dynamic shared mem size: %s", FormatWithCommas(static_cast<uint64_t>(val_size_bitmap_indexes)*sizeof(uint32_t)).c_str());
		CUDA_SAFE_CALL(cudaMalloc((void **) &d_bitmaps_arr[gpu_id], conc_blocks_per_SM * num_SMs * val_size_bitmap * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMemset(d_bitmaps_arr[gpu_id], 0, conc_blocks_per_SM * num_SMs * val_size_bitmap * sizeof(uint32_t)));

		/*initialize the bitmap states*/
		//		uint32_t *d_bitmap_states;
		CUDA_SAFE_CALL(cudaMalloc((void **) &d_bitmap_states_arr[gpu_id], num_SMs * conc_blocks_per_SM * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMemset(d_bitmap_states_arr[gpu_id], 0, num_SMs * conc_blocks_per_SM * sizeof(uint32_t)));

		/*vertex count for sequential block execution*/
		CUDA_SAFE_CALL(cudaMalloc((void **) &d_vertex_count_arr[gpu_id], sizeof(uint32_t)));
		//cudaDeviceSynchronize();
	}
#endif

#if defined(UNIFIED_MEM)
	log_info("adopt unified memory on GPU for Pascal+");
	cudaMallocManaged(&edge_dst, (edgemax+16) * sizeof(uint32_t));
#else
	edge_dst = static_cast<int *>(malloc(sizeof(int) * static_cast<uint64_t>(edgemax + 16)));
#endif

#ifdef MULTI_GPU
	log_info("multi-gpu advise edge_dst as read-mostly");

	for(auto i = 0; i< num_of_gpus; i++) {
		cudaMemAdvise(edge_dst, (edgemax+16) * sizeof(uint32_t), cudaMemAdviseSetReadMostly ,i);
	}
#endif

	string dst_v_file_name = dir + string("/b_adj.bin");
	auto dst_v_fd = open(dst_v_file_name.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
	int *buffer = (int *) mmap(0, static_cast<uint64_t>(edgemax) * 4u, PROT_READ, MAP_PRIVATE, dst_v_fd, 0);

	// prefix sum
	node_off[0] = 0;
	for (auto i = 0; i < nodemax; i++) {
		node_off[i + 1] = node_off[i] + degree[i];
	}

	auto end = high_resolution_clock::now();
	log_info("malloc, and sequential-scan time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);
	// load dst vertices into the array
#pragma omp parallel for schedule(dynamic, 1000)
	for (auto i = 0; i < nodemax; i++) {
		// copy to the high memory bandwidth mem
		for (uint64_t offset = node_off[i]; offset < node_off[i + 1]; offset++) {
			edge_dst[offset] = buffer[offset];
		}
#ifdef LEGACY
		if (degree[i] > 0) {
			adj_file.read(reinterpret_cast<char *>(&edge_dst[node_off[i]]), degree[i] * sizeof(int));
		}
#endif
		// inclusive
		degree[i]++;
	}
	munmap(buffer, static_cast<uint64_t>(edgemax) * 4u);

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
				cout << "Edges not sorted in increasing id order!\nThe program may not run properly!\n";
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
			[&ofs](pair<int, int> my_pair) {ofs << "n " << my_pair.second << " " << my_pair.first << "\n";});
	auto end2 = high_resolution_clock::now();
	log_info("non-cores output time: %.3lf s", duration_cast<milliseconds>(end2 - end).count() / 1000.0);
}

