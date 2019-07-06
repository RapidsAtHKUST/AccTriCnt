#include <atomic>
#include "../scan_xp_multi_gpu_multi_pass_common.cu" // copy the common codes and paste here, just a pre-processing by compiler

#define ALLOWED_MAX_PASS_NUM (10)
#define TASK_AMPLIFY_FACTOR (8)

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

#ifndef UNIFIED_MEM
	assert(false);
#endif
	// 1st: init the vid ranges for the multi-gpu vertex range assignment
	auto num_of_gpus = 1;
	cudaGetDeviceCount(&num_of_gpus);

	auto dynamic_task_queue_size = num_of_gpus * TASK_AMPLIFY_FACTOR;
	auto vid_range = ComputeVertexRangeAccumulatedFilteredDegree(g, dynamic_task_queue_size);
	for (auto i = 0; i < vid_range.size() - 1; i++) {
		if (vid_range[i] + 16 >= vid_range[i + 1]) {
			vid_range[i + 1] = std::min<int32_t>(vid_range[i] + 16, g->nodemax);
		}
		if (vid_range[i + 1] == g->nodemax) {
			dynamic_task_queue_size = i + 1;
			break;
		}
	}

	log_info("show the several %d task range", num_of_gpus);
	for (auto i = 0; i < num_of_gpus; i++) {
		log_info("[%s: %s)", FormatWithCommas(vid_range[i]).c_str(), FormatWithCommas(vid_range[i+1]).c_str());
	}log_info("... [%s: %s)", FormatWithCommas(vid_range[dynamic_task_queue_size-1]).c_str(), FormatWithCommas(vid_range[dynamic_task_queue_size]).c_str());

	log_info("num of tasks: %d", dynamic_task_queue_size);

	// 2nd: submit tasks
#if defined(USE_BITMAP_KERNEL) && defined(WARP_PER_VERTEX)
	uint32_t block_size = 128;
	const uint32_t TITAN_XP_WARP_SIZE = 32;
	dim3 t_dimension(WARP_SIZE,block_size/TITAN_XP_WARP_SIZE); /*2-D*/
#elif defined(USE_BITMAP_KERNEL)
	uint32_t block_size = 32;
	dim3 t_dimension(block_size);
#endif

	// for multi-pass range computation
	vector<uint32_t> multi_pass_task_prefix_sum(g->nodemax + 1);
	multi_pass_task_prefix_sum[0] = 0;
	for (auto u = 0; u < g->nodemax; u++) {
		multi_pass_task_prefix_sum[u + 1] = multi_pass_task_prefix_sum[u] + g->node_off[u + 1] - g->node_off[u];
	}

	std::atomic_int* dynamic_task_idx_arr = (std::atomic_int *) malloc(ALLOWED_MAX_PASS_NUM * sizeof(std::atomic_int));
	for (int i = 0; i < ALLOWED_MAX_PASS_NUM; i++) {
		dynamic_task_idx_arr[i] = 0;
	}

	// 3rd: offload set-intersection workloads to GPU
	// currently GPUs are required to be the same to make our dynamic scheduling work for the multi-pass
#pragma omp parallel num_threads(num_of_gpus)
	{
		auto gpu_id = omp_get_thread_num();
		cudaSetDevice(gpu_id);

#if defined(USE_BITMAP_KERNEL)
		extern uint32_t ** d_bitmaps_arr;
		extern uint32_t ** d_vertex_count_arr;
		extern uint32_t ** d_bitmap_states_arr;
		uint32_t * d_bitmaps = d_bitmaps_arr[gpu_id];
		uint32_t * d_vertex_count = d_vertex_count_arr[gpu_id];
		uint32_t * d_bitmap_states = d_bitmap_states_arr[gpu_id];

		/*get the maximal number of threads in an SM*/
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, gpu_id); /*currently 0th device*/
		uint32_t max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
		uint32_t num_SMs = prop.multiProcessorCount;

		uint32_t conc_blocks_per_SM = max_threads_per_SM / block_size; /*assume regs are not limited*/
		/*initialize the bitmaps*/
		const uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/
		const uint32_t val_size_bitmap = (g->nodemax + elem_bits - 1) / elem_bits;
		const uint32_t val_size_bitmap_indexes = (val_size_bitmap + BITMAP_SCALE - 1) / BITMAP_SCALE;

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
		/*vertex count for sequential block execution*/
#endif
		vector<uint32_t> multi_pass_range(num_of_pass + 1);
		multi_pass_range[0] = 0;
		for (auto i = 1; i < num_of_pass; i++) {
			multi_pass_range[i] = std::lower_bound(std::begin(multi_pass_task_prefix_sum),
					std::end(multi_pass_task_prefix_sum), multi_pass_task_prefix_sum[g->nodemax] / num_of_pass * i)
					- std::begin(multi_pass_task_prefix_sum);
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

		while (true) {
			auto dynamic_task_idx = dynamic_task_idx_arr[0].fetch_add(1, std::memory_order_seq_cst);
			if (dynamic_task_idx >= dynamic_task_queue_size) {
				break;
			}
			uint32_t num_blocks_to_process = vid_range[dynamic_task_idx + 1] - vid_range[dynamic_task_idx];
			auto start_vid = vid_range[dynamic_task_idx];
			auto start_pass_id = 0;

			// u < v property
			while (multi_pass_range[start_pass_id + 1] < start_vid) {
				start_pass_id++;
			}
			for (auto pass_id = start_pass_id; pass_id < num_of_pass; pass_id++) {
#if defined(USE_BITMAP_KERNEL)
				cudaMemcpy(d_vertex_count, &start_vid, sizeof(uint32_t), cudaMemcpyHostToDevice);
#endif

#if	defined(USE_BITMAP_KERNEL) && defined(WARP_PER_VERTEX) && defined(UNIFIED_MEM) && defined(BITMAP_1D)
				bool thread_local is_first = true;
				if(is_first && gpu_id ==0) {
					log_info("use bitmap 1D...");
					is_first = false;
				}
				set_intersection_GPU_bitmap_warp_per_vertex_1D<<<num_blocks_to_process, t_dimension>>>(
						g->node_off, g->edge_dst, d_bitmaps, d_bitmap_states, d_vertex_count, conc_blocks_per_SM, g->common_node_num, g->nodemax,
						multi_pass_range[pass_id], multi_pass_range[pass_id+1]);
#elif defined(USE_BITMAP_KERNEL) && defined(WARP_PER_VERTEX) && defined(UNIFIED_MEM)
				set_intersection_GPU_bitmap_warp_per_vertex<<<num_blocks_to_process, t_dimension, val_size_bitmap_indexes*sizeof(uint32_t)>>>(
						g->node_off, g->edge_dst, d_bitmaps, d_bitmap_states, d_vertex_count, conc_blocks_per_SM, g->common_node_num, g->nodemax,
						multi_pass_range[pass_id], multi_pass_range[pass_id+1]);
#elif defined(USE_BITMAP_KERNEL) && defined(UNIFIED_MEM)
				set_intersection_GPU_bitmap<<<num_blocks_to_process, t_dimension, val_size_bitmap_indexes*sizeof(uint32_t)>>>(g->node_off, g->edge_dst,
						d_bitmaps, d_bitmap_states, d_vertex_count, conc_blocks_per_SM, g->common_node_num, g->nodemax,
						multi_pass_range[pass_id], multi_pass_range[pass_id+1]);
#elif defined(USE_HYBRID_KERNELS)
				auto skew_ratio = 50;
				auto num_warps = 4;
				auto warp_size = 32;
				auto dim_size = num_warps * warp_size;
				dim3 t_dimension_warp(warp_size, num_warps);
				set_intersection_GPU_shared_merge_filtered<<<num_blocks_to_process, t_dimension_warp, dim_size*2*sizeof(uint32_t)>>>(g->node_off, g->edge_dst, g->common_node_num, start_vid,
						multi_pass_range[pass_id], multi_pass_range[pass_id+1], skew_ratio);
				set_intersection_GPU_shared_galloping_filtered<<<num_blocks_to_process, t_dimension>>>(g->node_off, g->edge_dst, g->common_node_num, start_vid,
						multi_pass_range[pass_id], multi_pass_range[pass_id+1], skew_ratio);
#elif defined(UNIFIED_MEM)
				set_intersection_GPU_shared<<<num_blocks_to_process, t_dimension>>>(g->node_off, g->edge_dst, g->common_node_num, start_vid,
						multi_pass_range[pass_id], multi_pass_range[pass_id+1]);
#else
				assert(false);
#endif
			}
			cudaDeviceSynchronize();
		}
		cudaEventRecord(cuda_end);

		cudaEventSynchronize(cuda_start);
		cudaEventSynchronize(cuda_end);
		cudaEventElapsedTime(&time_GPU, cuda_start, cuda_end);
#pragma omp critical
		log_info("CUDA Kernel Time: %.3lf ms", time_GPU);
		gpuErrchk(cudaPeekAtLastError());

		// copy back the intersection cnt
		cudaDeviceSynchronize();	// ensure the kernel execution finished
		auto end = high_resolution_clock::now();
#pragma omp critical
		log_info("CUDA kernel lauch cost in GPU %d: %.3lf s", gpu_id, duration_cast<milliseconds>(end - start).count() / 1000.0);
	}

	// 4th: join the coroutine, assign the remaining intersection-count values
	my_coroutine.join();
	PostComputeCoreChecking(this, g, min_u_, epsilon_);

	free(dynamic_task_idx_arr);
}
