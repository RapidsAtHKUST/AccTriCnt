#include "../cuda_utils/set_inter_device_functions.cuh"

#include "../scan_xp_common.cu"    // copy the common codes and paste here, just a pre-processing by compiler

#ifndef TILE_SRC
#define TILE_SRC (4)
#endif

#ifndef TILE_DST
#define TILE_DST (8)
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef SHARED_MEM_SIZE
#define SHARED_MEM_SIZE 64
#endif

/*for bitmap-based set intersection*/
#define BITMAP_SCALE_LOG (9)
#define BITMAP_SCALE (1<<BITMAP_SCALE_LOG)  /*#bits in the first-level bitmap indexed by 1 bit in the second-level bitmap*/

#define INIT_INTERSECTION_CNT (2)

__global__ void set_intersection_GPU_shared(uint32_t *d_offsets, /*card: |V|+1*/
int32_t *d_dsts, /*card: 2*|E|*/
int32_t *d_intersection_count_GPU, uint32_t pass_range_beg, uint32_t pass_range_end) /*card: 2*|E|*/
{
	const uint32_t tid = threadIdx.x * blockDim.y + threadIdx.y; // in [0, 32)
	const uint32_t u = blockIdx.x;

	const uint32_t off_u_beg = d_offsets[u];
	const uint32_t off_u_end = d_offsets[u + 1];

#if defined(BASELINE)
	for (uint32_t off_u_iter = off_u_beg + tid; off_u_iter < off_u_end; off_u_iter += 32) {
		auto v = d_dsts[off_u_iter];
		if (u > v)
		continue; /*skip when u > v*/
		if(v >= pass_range_beg && v < pass_range_end)
		{
			d_intersection_count_GPU[off_u_iter] = 2 + ComputeCNNaiveStdMergeDevice(d_offsets, d_dsts, u, v);
		}
	}
#elif defined(BASELINE_HYBRID)
	for (uint32_t off_u_iter = off_u_beg + tid; off_u_iter < off_u_end; off_u_iter += 32) {
		auto v = d_dsts[off_u_iter];
		if (u > v)
		continue; /*skip when u > v*/
		if(v >= pass_range_beg && v < pass_range_end)
		{
			d_intersection_count_GPU[off_u_iter] = 2 + ComputeCNHybridDevice(d_offsets, d_dsts, u, v);
		}
	}
#else
	__shared__ int32_t u_neis[SHARED_MEM_SIZE], v_neis[SHARED_MEM_SIZE];
	/*traverse all the neighbors(destination nodes)*/
	for (uint32_t off_u_iter = d_offsets[u]; off_u_iter < off_u_end; off_u_iter++) {
		const int32_t v = d_dsts[off_u_iter];

		uint32_t private_count = 0;
		if (u > v || v < pass_range_beg || v >= pass_range_end)
			continue; /*skip when u > v*/
		uint32_t off_u = off_u_beg;

		uint32_t off_v_beg = d_offsets[v];
		uint32_t off_v = off_v_beg;
		uint32_t off_v_end = d_offsets[v + 1];

		// first-time load
		for (auto i = tid; i < SHARED_MEM_SIZE; i += WARP_SIZE) {
			if (off_u + i < off_u_end)
				u_neis[i] = d_dsts[off_u + i];
		}
		for (auto i = tid; i < SHARED_MEM_SIZE; i += WARP_SIZE) {
			if (off_v + i < off_v_end)
				v_neis[i] = d_dsts[off_v + i];
		}

		while (true) {
			// commit 32 comparisons
			uint32_t off_u_local = threadIdx.x + off_u; /*A[0-3]*/
			uint32_t off_v_local = threadIdx.y + off_v; /*B[0-7]*/

			// 1st: all-pairs comparisons
			uint32_t elem_src =
					(off_u_local < off_u_end) ? u_neis[(off_u_local - off_u_beg) % SHARED_MEM_SIZE] : (UINT32_MAX);
			uint32_t elem_dst =
					(off_v_local < off_v_end) ? v_neis[(off_v_local - off_v_beg) % SHARED_MEM_SIZE] : (UINT32_MAX - 1);
			if (elem_src == elem_dst)
				private_count++;

			// 2nd: advance by 4 elements in A or 8 elements in B
			elem_src =
					(off_u + TILE_SRC - 1 < off_u_end) ?
							u_neis[(off_u + TILE_SRC - 1 - off_u_beg) % SHARED_MEM_SIZE] : (UINT32_MAX);
			elem_dst =
					(off_v + TILE_DST - 1 < off_v_end) ?
							v_neis[(off_v + TILE_DST - 1 - off_v_beg) % SHARED_MEM_SIZE] : (UINT32_MAX);
			// check whether to exit
			if (elem_src == UINT32_MAX && elem_dst == UINT32_MAX)
				break;

			if (elem_src < elem_dst) {
				off_u += TILE_SRC;
				if ((off_u - off_u_beg) % SHARED_MEM_SIZE == 0) {
					for (auto i = tid; i < SHARED_MEM_SIZE; i += WARP_SIZE) {
						if (off_u + i < off_u_end)
							u_neis[i] = d_dsts[off_u + i];
					}
				}
			} else {
				off_v += TILE_DST;
				if ((off_v - off_v_beg) % SHARED_MEM_SIZE == 0) {
					for (auto i = tid; i < SHARED_MEM_SIZE; i += WARP_SIZE) {
						if (off_v + i < off_v_end)
							v_neis[i] = d_dsts[off_v + i];
					}
				}
			}
		}
		/*single warp reduction*/
		for (int offset = 16; offset > 0; offset >>= 1)
			private_count += __shfl_down(private_count, offset);
		if (tid == 0)
			d_intersection_count_GPU[off_u_iter] = 2 + private_count;
	}
#endif
}

__global__ void set_intersection_GPU_bitmap(uint32_t *d_offsets, /*card: |V|+1*/
int32_t *d_dsts, /*card: 2*|E|*/
uint32_t *d_bitmaps, /*the global bitmaps*/
uint32_t *d_bitmap_states, /*recording the usage of the bitmaps on the SM*/
uint32_t *vertex_count, /*for sequential block execution*/
uint32_t conc_blocks_per_SM, /*#concurrent blocks per SM*/
int32_t *d_intersection_count_GPU, uint32_t pass_range_beg, uint32_t pass_range_end) /*card: 2*|E|*/
{
	const uint32_t tid = threadIdx.x;
	const uint32_t tnum = blockDim.x;
	const uint32_t num_nodes = gridDim.x; /*#nodes=#blocks*/
	const uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/
	const uint32_t val_size_bitmap = (num_nodes + elem_bits - 1) / elem_bits;
	const uint32_t val_size_bitmap_indexes = (val_size_bitmap + BITMAP_SCALE - 1) >> BITMAP_SCALE_LOG;

	__shared__ uint32_t intersection_count;
	__shared__ uint32_t node_id, sm_id, bitmap_ptr;
	__shared__ uint32_t start_src, end_src, start_src_in_bitmap, end_src_in_bitmap;

	extern __shared__ uint32_t bitmap_indexes[];

	if (tid == 0) {
		node_id = atomicAdd(vertex_count, 1); /*get current vertex id*/
		start_src = d_offsets[node_id];
		end_src = d_offsets[node_id + 1];
		start_src_in_bitmap = d_dsts[start_src] / elem_bits;
		end_src_in_bitmap = (start_src == end_src) ? d_dsts[start_src] / elem_bits : d_dsts[end_src - 1] / elem_bits;
		intersection_count = INIT_INTERSECTION_CNT;
	} else if (tid == tnum - 1) {
		uint32_t temp = 0;
		asm("mov.u32 %0, %smid;" : "=r"(sm_id) );
		/*get current SM*/
		while (atomicCAS(&d_bitmap_states[sm_id * conc_blocks_per_SM + temp], 0, 1) != 0)
			temp++;
		bitmap_ptr = temp;
	}
	/*initialize the 2-level bitmap*/
	for (uint32_t idx = tid; idx < val_size_bitmap_indexes; idx += tnum)
		bitmap_indexes[idx] = 0;
	__syncthreads();

	uint32_t *bitmap = &d_bitmaps[val_size_bitmap * (conc_blocks_per_SM * sm_id + bitmap_ptr)];

	/*construct the source node neighbor bitmap*/
	for (uint32_t idx = start_src + tid; idx < end_src; idx += tnum) {
		uint32_t src_nei = d_dsts[idx];
		const uint32_t src_nei_val = src_nei / elem_bits;
		atomicOr(&bitmap[src_nei_val], (0b1 << (src_nei & (elem_bits - 1)))); /*setting the bitmap*/
		atomicOr(&bitmap_indexes[src_nei_val >> BITMAP_SCALE_LOG],
				(0b1 << ((src_nei >> BITMAP_SCALE_LOG) & (elem_bits - 1)))); /*setting the bitmap index*/
	}

	/*loop the neighbors*/
	for (uint32_t idx = start_src; idx < end_src; idx++) {
		__syncthreads();
		uint32_t private_count = 0;
		uint32_t src_nei = d_dsts[idx];
		if (src_nei < node_id || src_nei < pass_range_beg || src_nei >= pass_range_end)
			continue;

		uint32_t start_dst = d_offsets[src_nei];
		uint32_t end_dst = d_offsets[src_nei + 1];
		for (uint32_t dst_idx = start_dst + tid; dst_idx < end_dst; dst_idx += tnum) {
			uint32_t dst_nei = d_dsts[dst_idx];
			const uint32_t dst_nei_val = dst_nei / elem_bits;
			if ((bitmap_indexes[dst_nei_val >> BITMAP_SCALE_LOG] >> ((dst_nei >> BITMAP_SCALE_LOG) & (elem_bits - 1)))
					& 0b1 == 1)
				if ((bitmap[dst_nei_val] >> (dst_nei & (elem_bits - 1))) & 0b1 == 1)
					private_count++;
		}

		private_count += __shfl_down(private_count, 16);
		private_count += __shfl_down(private_count, 8);
		private_count += __shfl_down(private_count, 4);
		private_count += __shfl_down(private_count, 2);
		private_count += __shfl_down(private_count, 1);
		if ((tid & 31) == 0)
			atomicAdd(&intersection_count, private_count);
		__syncthreads();

		if (tid == 0) {
			d_intersection_count_GPU[idx] = intersection_count;
			intersection_count = INIT_INTERSECTION_CNT;
		}
	}

	/*clean the bitmap*/
	if (end_src_in_bitmap - start_src_in_bitmap + 1 <= end_src - start_src) {
		for (uint32_t idx = start_src_in_bitmap + tid; idx <= end_src_in_bitmap; idx += tnum) {
			bitmap[idx] = 0;
		}
	} else {
		for (uint32_t idx = start_src + tid; idx < end_src; idx += tnum) {
			uint32_t src_nei = d_dsts[idx];
			bitmap[src_nei / elem_bits] = 0;
		}
	}
	__syncthreads();

	/*release the bitmap lock*/
	if (tid == 0)
		atomicCAS(&d_bitmap_states[sm_id * conc_blocks_per_SM + bitmap_ptr], 1, 0);
}

__global__ void set_intersection_GPU_bitmap_warp_per_vertex(uint32_t *d_offsets, /*card: |V|+1*/
int32_t *d_dsts, /*card: 2*|E|*/
uint32_t *d_bitmaps, /*the global bitmaps*/
uint32_t *d_bitmap_states, /*recording the usage of the bitmaps on the SM*/
uint32_t *vertex_count, /*for sequential block execution*/
uint32_t conc_blocks_per_SM, /*#concurrent blocks per SM*/
int32_t *d_intersection_count_GPU, uint32_t pass_range_beg, uint32_t pass_range_end) /*card: 2*|E|*/
{
	const uint32_t tid = threadIdx.x + blockDim.x * threadIdx.y; /*threads in a warp are with continuous threadIdx.x */
	const uint32_t tnum = blockDim.x * blockDim.y;
	const uint32_t num_nodes = gridDim.x; /*#nodes=#blocks*/
	const uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/
	const uint32_t val_size_bitmap = (num_nodes + elem_bits - 1) / elem_bits;
	const uint32_t val_size_bitmap_indexes = (val_size_bitmap + BITMAP_SCALE - 1) >> BITMAP_SCALE_LOG;

//	__shared__ uint32_t intersection_count;
	__shared__ uint32_t node_id, sm_id, bitmap_ptr;
	__shared__ uint32_t start_src, end_src, start_src_in_bitmap, end_src_in_bitmap;

	extern __shared__ uint32_t bitmap_indexes[];

	if (tid == 0) {
		node_id = atomicAdd(vertex_count, 1); /*get current vertex id*/
		start_src = d_offsets[node_id];
		end_src = d_offsets[node_id + 1];
		start_src_in_bitmap = d_dsts[start_src] / elem_bits;
		end_src_in_bitmap = (start_src == end_src) ? d_dsts[start_src] / elem_bits : d_dsts[end_src - 1] / elem_bits;
//		intersection_count = 0;
	} else if (tid == tnum - 1) {
		uint32_t temp = 0;
		asm("mov.u32 %0, %smid;" : "=r"(sm_id) );
		/*get current SM*/
		while (atomicCAS(&d_bitmap_states[sm_id * conc_blocks_per_SM + temp], 0, 1) != 0)
			temp++;
		bitmap_ptr = temp;
	}
	/*initialize the 2-level bitmap*/
	for (uint32_t idx = tid; idx < val_size_bitmap_indexes; idx += tnum)
		bitmap_indexes[idx] = 0;
	__syncthreads();

	uint32_t *bitmap = &d_bitmaps[val_size_bitmap * (conc_blocks_per_SM * sm_id + bitmap_ptr)];

	/*construct the source node neighbor bitmap*/
	for (uint32_t idx = start_src + tid; idx < end_src; idx += tnum) {
		uint32_t src_nei = d_dsts[idx];
		const uint32_t src_nei_val = src_nei / elem_bits;
		atomicOr(&bitmap[src_nei_val], (0b1 << (src_nei & (elem_bits - 1)))); /*setting the bitmap*/
		atomicOr(&bitmap_indexes[src_nei_val >> BITMAP_SCALE_LOG],
				(0b1 << ((src_nei >> BITMAP_SCALE_LOG) & (elem_bits - 1)))); /*setting the bitmap index*/
	}
	__syncthreads();

	/*loop the neighbors*/
	/* x dimension: warp-size
	 * y dimension: number of warps
	 * */
	for (uint32_t idx = start_src + threadIdx.y; idx < end_src; idx += blockDim.y) {
		/*each warp processes a node*/
		uint32_t private_count = 0;
		uint32_t src_nei = d_dsts[idx];
		if (src_nei < node_id || src_nei < pass_range_beg || src_nei >= pass_range_end)
			continue;

		uint32_t start_dst = d_offsets[src_nei];
		uint32_t end_dst = d_offsets[src_nei + 1];
		for (uint32_t dst_idx = start_dst + threadIdx.x; dst_idx < end_dst; dst_idx += blockDim.x) {
			uint32_t dst_nei = d_dsts[dst_idx];
			const uint32_t dst_nei_val = dst_nei / elem_bits;
			if ((bitmap_indexes[dst_nei_val >> BITMAP_SCALE_LOG] >> ((dst_nei >> BITMAP_SCALE_LOG) & (elem_bits - 1)))
					& 0b1 == 1)
				if ((bitmap[dst_nei_val] >> (dst_nei & (elem_bits - 1))) & 0b1 == 1)
					private_count++;
		}
		/*warp-wise reduction*/
		private_count += __shfl_down(private_count, 16);
		private_count += __shfl_down(private_count, 8);
		private_count += __shfl_down(private_count, 4);
		private_count += __shfl_down(private_count, 2);
		private_count += __shfl_down(private_count, 1);
		if (threadIdx.x == 0)
			d_intersection_count_GPU[idx] = private_count + INIT_INTERSECTION_CNT;
	}
	__syncthreads();

	/*clean the bitmap*/
	if (end_src_in_bitmap - start_src_in_bitmap + 1 <= end_src - start_src) {
		for (uint32_t idx = start_src_in_bitmap + tid; idx <= end_src_in_bitmap; idx += tnum) {
			bitmap[idx] = 0;
		}
	} else {
		for (uint32_t idx = start_src + tid; idx < end_src; idx += tnum) {
			uint32_t src_nei = d_dsts[idx];
			bitmap[src_nei / elem_bits] = 0;
		}
	}
	__syncthreads();

	/*release the bitmap lock*/
	if (tid == 0)
		atomicCAS(&d_bitmap_states[sm_id * conc_blocks_per_SM + bitmap_ptr], 1, 0);
}

void SCAN_XP::CheckCore(Graph *g) {
	auto start = high_resolution_clock::now();

	// co-processing with GPU: start a coroutine to compute the reverse offset, using binary-search
	std::thread my_coroutine([this, g]() {
        auto start = high_resolution_clock::now();

#pragma omp parallel for num_threads(thread_num_/2) schedule(dynamic, 60000)
			for (auto i = 0u; i < g->edgemax; i++) {
				// remove edge_src optimization, assuming task scheduling in FIFO-queue-mode
			static thread_local auto u = 0;
			u = FindSrc(g, u, i);
			auto v = g->edge_dst[i];
			if (u < v) {
				// reverse offset
				g->common_node_num[lower_bound(g->edge_dst + g->node_off[v], g->edge_dst +g->node_off[v + 1], u)- g->edge_dst] = i;
			}
		}
        auto end = high_resolution_clock::now();
        log_info("CPU corountine time: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);
		log_info("finish cross link");
	});

#if defined(USE_BITMAP_KERNEL) && defined(WARP_PER_VERTEX)
	uint32_t block_size = 128;
	const uint32_t TITAN_XP_WARP_SIZE = 32;
	dim3 t_dimension(WARP_SIZE,block_size/TITAN_XP_WARP_SIZE); /*2-D*/
#elif defined(USE_BITMAP_KERNEL)
	uint32_t block_size = 32;
	dim3 t_dimension(block_size);
#endif

	vector<uint32_t> task_prefix_sum(g->nodemax + 1);
	task_prefix_sum[0] = 0;
	for (auto u = 0; u < g->nodemax; u++) {
		task_prefix_sum[u + 1] = task_prefix_sum[u] + g->node_off[u + 1] - g->node_off[u];
	}

#if defined(USE_BITMAP_KERNEL)
	uint32_t *d_bitmaps, *d_vertex_count;
	/*get the maximal number of threads in an SM*/
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); /*currently 0th device*/
	uint32_t max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
	uint32_t num_SMs = prop.multiProcessorCount;

	uint32_t conc_blocks_per_SM = max_threads_per_SM / block_size; /*assume regs are not limited*/
	/*initialize the bitmaps*/
	const uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/
	const uint32_t val_size_bitmap = (g->nodemax + elem_bits - 1) / elem_bits;
	const uint32_t val_size_bitmap_indexes = (val_size_bitmap + BITMAP_SCALE - 1) / BITMAP_SCALE;

	cudaMalloc((void **) &d_bitmaps, conc_blocks_per_SM * num_SMs * val_size_bitmap * sizeof(uint32_t));
	cudaMemset(d_bitmaps, 0, conc_blocks_per_SM * num_SMs * val_size_bitmap * sizeof(uint32_t));

	/*initialize the bitmap states*/
	uint32_t *d_bitmap_states;
	cudaMalloc((void **) &d_bitmap_states, num_SMs * conc_blocks_per_SM * sizeof(uint32_t));
	cudaMemset(d_bitmap_states, 0, num_SMs * conc_blocks_per_SM * sizeof(uint32_t));

	/*vertex count for sequential block execution*/
	cudaMalloc((void **) &d_vertex_count, sizeof(uint32_t));
	cudaMemset(d_vertex_count, 0, sizeof(uint32_t));

	log_info("bitmap bytes: %s Bytes", FormatWithCommas(
					static_cast<uint64_t>(conc_blocks_per_SM) * num_SMs * val_size_bitmap * sizeof(uint32_t)).c_str());
	log_info("dynamic shared mem size: %s", FormatWithCommas(static_cast<uint64_t>(val_size_bitmap_indexes)*sizeof(uint32_t)).c_str());
	int num_of_pass = std::ceil(static_cast<double>(g->edgemax)* sizeof(uint32_t) / (static_cast<uint64_t>(12)*1024*1024*1024-500*1024*1024-
					static_cast<uint64_t>(conc_blocks_per_SM) * num_SMs * val_size_bitmap * sizeof(uint32_t)
					- static_cast<uint64_t>(g->nodemax)* sizeof(uint32_t)));
#else
	int num_of_pass = std::ceil(
			static_cast<double>(g->edgemax) * sizeof(uint32_t)
					/ (static_cast<uint64_t>(12) * 1024 * 1024 * 1024 - 500 * 1024 * 1024
							- -static_cast<uint64_t>(g->nodemax) * sizeof(uint32_t)));
	// compute all intersections, do not prune currently
	dim3 t_dimension(TILE_SRC, TILE_DST); /*2-D*/
#endif
	vector<uint32_t> multi_pass_range(num_of_pass + 1);
	multi_pass_range[0] = 0;
	for (auto i = 1; i < num_of_pass; i++) {
		multi_pass_range[i] = std::lower_bound(std::begin(task_prefix_sum), std::end(task_prefix_sum),
				task_prefix_sum[g->nodemax] / num_of_pass * i) - std::begin(task_prefix_sum);
	}
	multi_pass_range[num_of_pass] = g->nodemax;
#pragma omp single nowait
	for (auto i = 0; i < num_of_pass; i++) {
		log_info("[%s: %s)", FormatWithCommas(multi_pass_range[i]).c_str(), FormatWithCommas(multi_pass_range[i+1]).c_str());
	}

	cudaEvent_t cuda_start, cuda_end;
	cudaEventCreate(&cuda_start);
	cudaEventCreate(&cuda_end);

	float time_GPU;
	cudaEventRecord(cuda_start);

	for (auto pass_id = 0; pass_id < num_of_pass; pass_id++) {
#if defined(USE_BITMAP_KERNEL)
		cudaMemset(d_vertex_count, 0, sizeof(uint32_t));
#endif
#if defined(USE_BITMAP_KERNEL) && defined(WARP_PER_VERTEX)
		set_intersection_GPU_bitmap_warp_per_vertex<<<g->nodemax, t_dimension, val_size_bitmap_indexes*sizeof(uint32_t)>>>(
				g->node_off, g->edge_dst, d_bitmaps, d_bitmap_states, d_vertex_count, conc_blocks_per_SM, g->common_node_num,
				multi_pass_range[pass_id], multi_pass_range[pass_id+1]);
#elif defined(USE_BITMAP_KERNEL)
		set_intersection_GPU_bitmap<<<g->nodemax, t_dimension, val_size_bitmap_indexes*sizeof(uint32_t)>>>(g->node_off, g->edge_dst,
				d_bitmaps, d_bitmap_states, d_vertex_count, conc_blocks_per_SM, g->common_node_num,
				multi_pass_range[pass_id], multi_pass_range[pass_id+1]);
#else
		set_intersection_GPU_shared<<<g->nodemax, t_dimension>>>(g->node_off, g->edge_dst, g->common_node_num,
				multi_pass_range[pass_id], multi_pass_range[pass_id + 1]);
#endif
	}

	cudaEventRecord(cuda_end);

	cudaEventSynchronize(cuda_start);
	cudaEventSynchronize(cuda_end);
	cudaEventElapsedTime(&time_GPU, cuda_start, cuda_end);
	log_info("CUDA Kernel Time: %.3lf ms", time_GPU);
	gpuErrchk(cudaPeekAtLastError());

	cudaDeviceSynchronize();	// ensure the kernel execution finished

	auto end = high_resolution_clock::now();
	log_info("CUDA kernel lauch cost: %.3lf s", duration_cast<milliseconds>(end - start).count() / 1000.0);

	// 4th: join the coroutine, assign the remaining intersection-count values
	my_coroutine.join();
	PostComputeCoreChecking(this, g, min_u_, epsilon_);
}
