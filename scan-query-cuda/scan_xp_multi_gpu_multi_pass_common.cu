#include "stdio.h"
#include <cassert>
#include "cuda-utils/set_inter_device_functions.cuh"
#include "cuda-utils/cuda_util.cuh"

#include "scan_xp_common.cu"	// copy the common codes and paste here, just a pre-processing by compiler

#ifndef TILE_SRC
#define TILE_SRC (4)
#endif

#ifndef TILE_DST
#define TILE_DST (8)
#endif

#define WARP_SHFL_MASK (0xFFFFFFFF)

#ifndef WARP_SIZE
#define WARP_SIZE 32
#define WARP_MASK (31u)
#endif
#ifndef SHARED_MEM_SIZE
#define SHARED_MEM_SIZE 64
#endif

/*for bitmap-based set intersection*/
#define BITMAP_SCALE_LOG (9)
#define BITMAP_SCALE (1<<BITMAP_SCALE_LOG)  /*#bits in the first-level bitmap indexed by 1 bit in the second-level bitmap*/

#define INIT_INTERSECTION_CNT (2)

__global__ void set_intersection_GPU_shared_galloping_filtered(uint32_t *d_offsets, /*card: |V|+1*/
int32_t *d_dsts, /*card: 2*|E|*/
int32_t *d_intersection_count_GPU, /*card: 2*|E|*/
uint32_t d_vid_beg, uint32_t pass_range_beg, uint32_t pass_range_end, int skew_ratio) {
	const uint32_t tid = threadIdx.x * blockDim.y + threadIdx.y; // in [0, 32)
	const uint32_t u = blockIdx.x + d_vid_beg;
	const uint32_t off_u_beg_const = d_offsets[u];
	const uint32_t off_u_end_const = d_offsets[u + 1];
	for (uint32_t off_u_iter = off_u_beg_const + tid; off_u_iter < off_u_end_const; off_u_iter += 32) {
		auto v = d_dsts[off_u_iter];
		if (v >= pass_range_beg && v < pass_range_end) {
			if (u > v)
				continue; /*skip when u > v*/
			uint32_t off_u_beg = off_u_beg_const;
			uint32_t off_u = off_u_beg_const;
			uint32_t off_u_end = off_u_end_const;

			uint32_t off_v_beg = d_offsets[v];
			uint32_t off_v_end = d_offsets[v + 1];
			uint32_t off_v = off_v_beg;
			if (off_u_end - off_u > off_v_end - off_v) {
				swap(&off_u_beg, &off_v_beg);
				swap(&off_u, &off_v);
				swap(&off_u_end, &off_v_end);
			}
			if (skew_ratio * (off_u_end - off_u_beg) >= (off_v_end - off_v_beg)) {
				continue;
			}

// ComputeCNGallopingSingleDirDeviceFixed: must ensure d[u] < d[v]
			d_intersection_count_GPU[off_u_iter] = INIT_INTERSECTION_CNT
					+ ComputeCNGallopingSingleDirDeviceFixed(d_offsets, d_dsts, off_u, off_v, off_u_end, off_v_end);
		}
	}
}

__global__ void set_intersection_GPU_shared_merge_filtered(uint32_t *d_offsets, /*card: |V|+1*/
int32_t *d_dsts, /*card: 2*|E|*/
int32_t *d_intersection_count_GPU, /*card: 2*|E|*/
uint32_t d_vid_beg, uint32_t pass_range_beg, uint32_t pass_range_end, int skew_ratio) {
	const uint32_t lane_id = threadIdx.x;
	const uint32_t warp_id = threadIdx.y;
	const uint32_t shared_mem_beg = warp_id * WARP_SIZE;
//	if (blockIdx.x == 0)
//		printf("Hello from thread %d, %d, %d\n", lane_id, warp_id, blockDim.y);
	const uint32_t u = blockIdx.x + d_vid_beg;
	const uint32_t off_u_beg_const = d_offsets[u];
	const uint32_t off_u_end_const = d_offsets[u + 1];

	extern __shared__ int32_t neis[];
	int32_t* u_neis = neis;
	int32_t* v_neis = &u_neis[blockDim.x * blockDim.y];
	/*traverse all the neighbors(destination nodes)*/
	for (uint32_t off_u_iter = off_u_beg_const + warp_id; off_u_iter < off_u_end_const; off_u_iter += blockDim.y) {
		const int32_t v = d_dsts[off_u_iter];
		if (v >= pass_range_beg && v < pass_range_end) {
			uint32_t private_count = 0;
			if (u > v)
				continue; /*skip when u > v*/
			uint32_t off_u_beg = off_u_beg_const;
			uint32_t off_u = off_u_beg_const;
			uint32_t off_u_end = off_u_end_const;

			uint32_t off_v_beg = d_offsets[v];
			uint32_t off_v_end = d_offsets[v + 1];
			uint32_t off_v = off_v_beg;
			if (off_u_end - off_u > off_v_end - off_v) {
				swap(&off_u_beg, &off_v_beg);
				swap(&off_u, &off_v);
				swap(&off_u_end, &off_v_end);
			}
			uint32_t tile_src;
			uint32_t tile_dst;
			if (skew_ratio * (off_u_end - off_u_beg) < (off_v_end - off_v_beg)) {
				continue;
			}
			if (4 * (off_u_end - off_u_beg) >= (off_v_end - off_v_beg)) {
				tile_src = 4;
				tile_dst = 8;
			} else if (8 * (off_u_end - off_u_beg) >= (off_v_end - off_v_beg)) {
				tile_src = 2;
				tile_dst = 16;
			} else {
				tile_src = 1;
				tile_dst = 32;
			}

// first-time load
			if (off_u + lane_id < off_u_end)
				u_neis[shared_mem_beg + lane_id] = d_dsts[off_u + lane_id];
			if (off_v + lane_id < off_v_end)
				v_neis[shared_mem_beg + lane_id] = d_dsts[off_v + lane_id];

			while (true) {
// commit 32 comparisons
				uint32_t off_u_local = (lane_id / tile_dst) + off_u; /*e.g. A[0-3]*/
				uint32_t off_v_local = (lane_id % tile_dst) + off_v; /*e.g. B[0-7]*/

// 1st: all-pairs comparisons
				uint32_t elem_src =
						(off_u_local < off_u_end) ?
								u_neis[shared_mem_beg + ((off_u_local - off_u_beg) % WARP_SIZE)] : (UINT32_MAX);
				uint32_t elem_dst =
						(off_v_local < off_v_end) ?
								v_neis[shared_mem_beg + ((off_v_local - off_v_beg) % WARP_SIZE)] : (UINT32_MAX - 1);
				if (elem_src == elem_dst)
					private_count++;

// 2nd: advance by 4 elements in A or 8 elements in B
				elem_src =
						(off_u + tile_src - 1 < off_u_end) ?
								u_neis[shared_mem_beg + ((off_u + tile_src - 1 - off_u_beg) % WARP_SIZE)] :
								(UINT32_MAX);
				elem_dst =
						(off_v + tile_dst - 1 < off_v_end) ?
								v_neis[shared_mem_beg + ((off_v + tile_dst - 1 - off_v_beg) % WARP_SIZE)] :
								(UINT32_MAX);
// check whether to exit
				if (elem_src == UINT32_MAX && elem_dst == UINT32_MAX)
					break;

				if (elem_src < elem_dst) {
					off_u += tile_src;
					if ((off_u - off_u_beg) % WARP_SIZE == 0) {
						if (off_u + lane_id < off_u_end)
							u_neis[shared_mem_beg + lane_id] = d_dsts[off_u + lane_id];
					}
				} else {
					off_v += tile_dst;
					if ((off_v - off_v_beg) % WARP_SIZE == 0) {
						if (off_v + lane_id < off_v_end)
							v_neis[shared_mem_beg + lane_id] = d_dsts[off_v + lane_id];

					}
				}
			}

			/*single warp reduction*/
			for (int offset = 16; offset > 0; offset >>= 1)
				private_count += __shfl_down_sync(WARP_SHFL_MASK, private_count, offset, WARP_SIZE);
			if (lane_id == 0)
				d_intersection_count_GPU[off_u_iter] = INIT_INTERSECTION_CNT + private_count;
		}
	}
}

#ifndef DEFAULT_SKEW_RAIO
#define DEFAULT_SKEW_RAIO (50)
#endif

__global__ void set_intersection_GPU_shared(uint32_t *d_offsets, /*card: |V|+1*/
int32_t *d_dsts, /*card: 2*|E|*/
int32_t *d_intersection_count_GPU, /*card: 2*|E|*/
uint32_t d_vid_beg, uint32_t pass_range_beg, uint32_t pass_range_end) {
	const uint32_t tid = threadIdx.x * blockDim.y + threadIdx.y; // in [0, 32)
	const uint32_t u = blockIdx.x + d_vid_beg;
	const uint32_t off_u_beg_const = d_offsets[u];
	const uint32_t off_u_end_const = d_offsets[u + 1];

#if defined(BASELINE)
	for (uint32_t off_u_iter = off_u_beg_const + tid; off_u_iter < off_u_end_const; off_u_iter += 32) {
		auto v = d_dsts[off_u_iter];
		if (u > v)
		continue; /*skip when u > v*/
		if(v >= pass_range_beg && v < pass_range_end)
		{
			d_intersection_count_GPU[off_u_iter] = INIT_INTERSECTION_CNT + ComputeCNNaiveStdMergeDevice(d_offsets, d_dsts, u, v);
		}
	}
#elif defined(BASELINE_HYBRID)
	for (uint32_t off_u_iter = off_u_beg_const + tid; off_u_iter < off_u_end_const; off_u_iter += 32) {
		auto v = d_dsts[off_u_iter];
		if (u > v)
		continue; /*skip when u > v*/
		if(v >= pass_range_beg && v < pass_range_end)
		{
			d_intersection_count_GPU[off_u_iter] = INIT_INTERSECTION_CNT + ComputeCNHybridDevice(d_offsets, d_dsts, u, v);
		}
	}
#else
	__shared__ int32_t u_neis[SHARED_MEM_SIZE], v_neis[SHARED_MEM_SIZE];
	/*traverse all the neighbors(destination nodes)*/
	for (uint32_t off_u_iter = off_u_beg_const; off_u_iter < off_u_end_const; off_u_iter++) {
		const int32_t v = d_dsts[off_u_iter];
		if (v >= pass_range_beg && v < pass_range_end) {
			if (u > v)
				continue; /*skip when u > v*/

// ensure d[u] < d[v]
			uint32_t off_u_beg = off_u_beg_const;
			uint32_t off_u = off_u_beg_const;
			uint32_t off_u_end = off_u_end_const;

			uint32_t off_v_beg = d_offsets[v];
			uint32_t off_v_end = d_offsets[v + 1];
			uint32_t off_v = off_v_beg;
			if (off_u_end - off_u > off_v_end - off_v) {
				swap(&off_u_beg, &off_v_beg);
				swap(&off_u, &off_v);
				swap(&off_u_end, &off_v_end);
			}

			if (DEFAULT_SKEW_RAIO * (off_u_end - off_u_beg) < (off_v_end - off_v_beg)) {
// Todo: Implement a warp-wise pivot-based set intersection
// single-dir-galloping-search pivot-based set intersection
				if (tid == 0)
					d_intersection_count_GPU[off_u_iter] = INIT_INTERSECTION_CNT
							+ ComputeCNGallopingSingleDirDeviceFixed(d_offsets, d_dsts, off_u, off_v, off_u_end,
									off_v_end);
			} else {
				uint32_t private_count = 0;
// first-time load
				for (auto i = tid; i < SHARED_MEM_SIZE; i += WARP_SIZE) {
					if (off_u + i < off_u_end)
						u_neis[i] = d_dsts[off_u + i];
				}
				for (auto i = tid; i < SHARED_MEM_SIZE; i += WARP_SIZE) {
					if (off_v + i < off_v_end)
						v_neis[i] = d_dsts[off_v + i];
				}
// block-wise merge-based set intersection
				while (true) {
// commit 32 comparisons
					uint32_t off_u_local = threadIdx.x + off_u; /*A[0-3]*/
					uint32_t off_v_local = threadIdx.y + off_v; /*B[0-7]*/

// 1st: all-pairs comparisons
					uint32_t elem_src =
							(off_u_local < off_u_end) ?
									u_neis[(off_u_local - off_u_beg) % SHARED_MEM_SIZE] : (UINT32_MAX);
					uint32_t elem_dst =
							(off_v_local < off_v_end) ?
									v_neis[(off_v_local - off_v_beg) % SHARED_MEM_SIZE] : (UINT32_MAX - 1);
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
					private_count += __shfl_down_sync(WARP_SHFL_MASK, private_count, offset, WARP_SIZE);
				if (tid == 0)
					d_intersection_count_GPU[off_u_iter] = 2 + private_count;
			}
		}
	}
#endif
}

__global__ void set_intersection_GPU_bitmap_warp_per_vertex(uint32_t *d_offsets, /*card: |V|+1*/
int32_t *d_dsts, /*card: 2*|E|*/
uint32_t *d_bitmaps, /*the global bitmaps*/
uint32_t *d_bitmap_states, /*recording the usage of the bitmaps on the SM*/
uint32_t *vertex_count, /*for sequential block execution*/
uint32_t conc_blocks_per_SM, /*#concurrent blocks per SM*/
int32_t *d_intersection_count_GPU, uint32_t num_nodes, uint32_t pass_range_beg, uint32_t pass_range_end) /*card: 2*|E|*/
{
	const uint32_t tid = threadIdx.x + blockDim.x * threadIdx.y; /*threads in a warp are with continuous threadIdx.x */
	const uint32_t tnum = blockDim.x * blockDim.y;
//	const uint32_t num_nodes = gridDim.x; /*#nodes=#blocks*/
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
		for (int offset = 16; offset > 0; offset >>= 1)
			private_count += __shfl_down_sync(WARP_SHFL_MASK, private_count, offset, WARP_SIZE);
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

__global__ void set_intersection_GPU_bitmap_warp_per_vertex_1D(uint32_t *d_offsets, /*card: |V|+1*/
int32_t *d_dsts, /*card: 2*|E|*/
uint32_t *d_bitmaps, /*the global bitmaps*/
uint32_t *d_bitmap_states, /*recording the usage of the bitmaps on the SM*/
uint32_t *vertex_count, /*for sequential block execution*/
uint32_t conc_blocks_per_SM, /*#concurrent blocks per SM*/
int32_t *d_intersection_count_GPU, uint32_t num_nodes, uint32_t pass_range_beg, uint32_t pass_range_end) /*card: 2*|E|*/
{
	const uint32_t tid = threadIdx.x + blockDim.x * threadIdx.y; /*threads in a warp are with continuous threadIdx.x */
	const uint32_t tnum = blockDim.x * blockDim.y;
	const uint32_t elem_bits = sizeof(uint32_t) * 8; /*#bits in a bitmap element*/
	const uint32_t val_size_bitmap = (num_nodes + elem_bits - 1) / elem_bits;

	__shared__ uint32_t node_id, sm_id, bitmap_ptr;
	__shared__ uint32_t start_src, end_src, start_src_in_bitmap, end_src_in_bitmap;

	if (tid == 0) {
		node_id = atomicAdd(vertex_count, 1); /*get current vertex id*/
		start_src = d_offsets[node_id];
		end_src = d_offsets[node_id + 1];
		start_src_in_bitmap = d_dsts[start_src] / elem_bits;
		end_src_in_bitmap = (start_src == end_src) ? d_dsts[start_src] / elem_bits : d_dsts[end_src - 1] / elem_bits;
	} else if (tid == tnum - 1) {
		uint32_t temp = 0;
		asm("mov.u32 %0, %smid;" : "=r"(sm_id) );
		/*get current SM*/
		while (atomicCAS(&d_bitmap_states[sm_id * conc_blocks_per_SM + temp], 0, 1) != 0)
			temp++;
		bitmap_ptr = temp;
	}
	__syncthreads();

	uint32_t *bitmap = &d_bitmaps[val_size_bitmap * (conc_blocks_per_SM * sm_id + bitmap_ptr)];

	/*construct the source node neighbor bitmap*/
	for (uint32_t idx = start_src + tid; idx < end_src; idx += tnum) {
		uint32_t src_nei = d_dsts[idx];
		const uint32_t src_nei_val = src_nei / elem_bits;
		atomicOr(&bitmap[src_nei_val], (0b1 << (src_nei & (elem_bits - 1)))); /*setting the bitmap*/
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

			if ((bitmap[dst_nei_val] >> (dst_nei & (elem_bits - 1))) & 0b1 == 1)
				private_count++;
		}
		/*warp-wise reduction*/
		for (int offset = 16; offset > 0; offset >>= 1)
			private_count += __shfl_down_sync(WARP_SHFL_MASK, private_count, offset, WARP_SIZE);
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

vector<int32_t> ComputeVertexRangeNaive(Graph* g, int num_of_gpus) {
	vector<int32_t> vid_range(num_of_gpus + 1);
	vid_range[0] = 0;

	// compute the range
	uint32_t accumulated_deg = 0;
	uint32_t tmp_idx = 1;
	for (auto i = 0; i < g->nodemax; i++) {
		accumulated_deg += g->node_off[i + 1] - g->node_off[i];
		if (accumulated_deg >= g->edgemax / num_of_gpus) {
			vid_range[tmp_idx] = i + 1;
			tmp_idx++;
			accumulated_deg = 0;
		}
	}
	vid_range[num_of_gpus] = g->nodemax;
	return vid_range;
}

vector<int32_t> ComputeVertexRangeAccumulatedFilteredDegree(Graph* g, int num_of_gpus) {
	auto start = high_resolution_clock::now();
	vector<int32_t> vid_range(num_of_gpus + 1);
	vid_range[0] = 0;

	vector<uint32_t> degree_sum_per_vertex_lst(g->nodemax);
	vector<uint64_t> prefix_sum(g->nodemax + 1);

	// 100 to avoid task too small, too much taking task overhead
#pragma omp parallel for schedule(dynamic, 100)
	for (auto u = 0; u < g->nodemax; u++) {
		auto it = std::lower_bound(g->edge_dst + g->node_off[u], g->edge_dst + g->node_off[u + 1], u);
		auto offset_beg = (it - (g->edge_dst + g->node_off[u])) + g->node_off[u];
//		degree_sum_per_vertex_lst[u] = 4 * (offset_beg - g->node_off[u]); // 4 for estimated access cost
		for (auto offset = offset_beg; offset < g->node_off[u + 1]; offset++) {
			auto v = g->edge_dst[offset];
			auto deg_v = g->node_off[v + 1] - g->node_off[v] + 2;	// 2 for hash table creation and clear
			degree_sum_per_vertex_lst[u] += deg_v;
		}
	}
	auto middle = high_resolution_clock::now();
	log_info("filtered accumulated degree cost : %.3lf s, Mem Usage: %s KB",
			duration_cast<milliseconds>(middle - start).count() / 1000.0, FormatWithCommas(getValue()).c_str());

	// currently sequential scan
	prefix_sum[0] = 0;
	for (auto u = 0; u < g->nodemax; u++) {
		prefix_sum[u + 1] = prefix_sum[u] + degree_sum_per_vertex_lst[u];
	}
	for (auto i = 1; i < num_of_gpus; i++) {
		vid_range[i] = std::lower_bound(std::begin(prefix_sum), std::end(prefix_sum),
				prefix_sum[g->nodemax] / num_of_gpus * i) - std::begin(prefix_sum);
	}
	vid_range[num_of_gpus] = g->nodemax;
	auto end = high_resolution_clock::now();
	log_info("task range init by filtered accumulated degree cost : %.3lf s, Mem Usage: %s KB",
			duration_cast<milliseconds>(end - start).count() / 1000.0, FormatWithCommas(getValue()).c_str());
	return vid_range;
}
