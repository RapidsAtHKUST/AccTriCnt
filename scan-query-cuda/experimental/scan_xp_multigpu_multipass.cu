#include "../scan_xp_multi_gpu_multi_pass_common.cu"

#define WARP_PER_VERTEX

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
	auto vid_range = ComputeVertexRangeAccumulatedFilteredDegree(g, num_of_gpus);

	for (auto i = 0; i < num_of_gpus; i++) {
		log_info("[%s: %s)", FormatWithCommas(vid_range[i]).c_str(), FormatWithCommas(vid_range[i+1]).c_str());
	}

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
	vector<uint32_t> task_prefix_sum(g->nodemax + 1);
	task_prefix_sum[0] = 0;
	for (auto u = 0; u < g->nodemax; u++) {
		task_prefix_sum[u + 1] = task_prefix_sum[u] + g->node_off[u + 1] - g->node_off[u];
	}

	// 3rd: offload set-intersection workloads to GPU
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

		auto val = vid_range[gpu_id];
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

		uint32_t num_blocks_to_process = vid_range[gpu_id + 1] - vid_range[gpu_id];
		auto start_pass_id = 0;

		// u < v property
		while (multi_pass_range[start_pass_id + 1] < vid_range[gpu_id]) {
			start_pass_id++;
		}
		for (auto pass_id = start_pass_id; pass_id < num_of_pass; pass_id++) {
#if defined(USE_BITMAP_KERNEL)
			cudaMemcpy(d_vertex_count, &val, sizeof(uint32_t), cudaMemcpyHostToDevice);
#endif

#if	defined(USE_BITMAP_KERNEL) && defined(WARP_PER_VERTEX) && defined(UNIFIED_MEM) && defined(BITMAP_1D)
			set_intersection_GPU_bitmap_warp_per_vertex_1D<<<num_blocks_to_process, t_dimension>>>(
					g->node_off, g->edge_dst, d_bitmaps, d_bitmap_states, d_vertex_count, conc_blocks_per_SM, g->common_node_num, g->nodemax,
					multi_pass_range[pass_id], multi_pass_range[pass_id+1]);
#elif defined(USE_BITMAP_KERNEL) && defined(UNIFIED_MEM)
			set_intersection_GPU_bitmap_warp_per_vertex<<<num_blocks_to_process, t_dimension, val_size_bitmap_indexes*sizeof(uint32_t)>>>(
					g->node_off, g->edge_dst, d_bitmaps, d_bitmap_states, d_vertex_count, conc_blocks_per_SM, g->common_node_num, g->nodemax,
					multi_pass_range[pass_id], multi_pass_range[pass_id+1]);
#elif defined(USE_HYBRID_KERNELS)
			auto skew_ratio = 50;
			auto num_warps = 4;
			auto warp_size = 32;
			auto dim_size = num_warps * warp_size;
			dim3 t_dimension_warp(warp_size, num_warps);
			set_intersection_GPU_shared_merge_filtered<<<num_blocks_to_process, t_dimension_warp, dim_size*2*sizeof(uint32_t)>>>(g->node_off, g->edge_dst, g->common_node_num, vid_range[gpu_id],
					multi_pass_range[pass_id], multi_pass_range[pass_id+1], skew_ratio);
			dim3 t_dimension_threading(32);
			set_intersection_GPU_shared_galloping_filtered<<<num_blocks_to_process, t_dimension_threading>>>(g->node_off, g->edge_dst, g->common_node_num, vid_range[gpu_id],
					multi_pass_range[pass_id], multi_pass_range[pass_id+1], skew_ratio);
#elif defined(UNIFIED_MEM)
			set_intersection_GPU_shared<<<num_blocks_to_process, t_dimension>>>(g->node_off, g->edge_dst, g->common_node_num, vid_range[gpu_id],
					multi_pass_range[pass_id], multi_pass_range[pass_id+1]);
#else
			assert(false);
#endif
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
}
