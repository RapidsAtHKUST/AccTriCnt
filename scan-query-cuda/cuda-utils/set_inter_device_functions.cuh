#pragma once

#include <cstdint>

__device__ void swap(uint32_t* left, uint32_t* right) {
	uint32_t tmp = *left;
	*left = *right;
	*right = tmp;
}

template<typename T>
__device__ uint32_t BinarySearchForGallopingSearchDevice(const T *array, uint32_t offset_beg, uint32_t offset_end,
		int val) {
	while (offset_end - offset_beg >= 1) {
		auto mid = static_cast<uint32_t>((static_cast<unsigned long>(offset_beg) + offset_end) / 2);
		if (array[mid] == val) {
			return mid;
		} else if (array[mid] < val) {
			offset_beg = mid + 1;
		} else {
			offset_end = mid;
		}
	}

	// linear search fallback
//	for (auto offset = offset_beg; offset < offset_end; offset++) {
//		if (array[offset] >= val) {
//			return offset;
//		}
//	}
	return offset_end;
}

template<typename T>
__device__ uint32_t GallopingSearchDevice(T *array, uint32_t offset_beg, uint32_t offset_end, int val) {
	if (array[offset_end - 1] < val) {
		return offset_end;
	}
	// galloping
	if (array[offset_beg] >= val) {
		return offset_beg;
	}
	if (array[offset_beg + 1] >= val) {
		return offset_beg + 1;
	}
	if (array[offset_beg + 2] >= val) {
		return offset_beg + 2;
	}

	auto jump_idx = 4u;
	while (true) {
		auto peek_idx = offset_beg + jump_idx;
		if (peek_idx >= offset_end) {
			return BinarySearchForGallopingSearchDevice(array, (jump_idx >> 1) + offset_beg + 1, offset_end, val);
		}
		if (array[peek_idx] < val) {
			jump_idx <<= 1;
		} else {
			return array[peek_idx] == val ?
					peek_idx :
					BinarySearchForGallopingSearchDevice(array, (jump_idx >> 1) + offset_beg + 1, peek_idx + 1, val);
		}
	}
}

template<typename T>
__device__ int ComputeCNNaiveStdMergeDevice(uint32_t *node_off, T *edge_dst, int u, int v) {
	auto cn_count = 0;
	auto offset_nei_u = node_off[u], offset_nei_v = node_off[v];
	auto off_u_end = node_off[u + 1], off_v_end = node_off[v + 1];
	while (true) {
		if (edge_dst[offset_nei_u] < edge_dst[offset_nei_v]) {
			++offset_nei_u;
			if (offset_nei_u >= off_u_end) {
				return cn_count;
			}
		} else if (edge_dst[offset_nei_u] > edge_dst[offset_nei_v]) {
			++offset_nei_v;
			if (offset_nei_v >= off_v_end) {
				return cn_count;
			}
		} else {
			cn_count++;
			++offset_nei_u;
			++offset_nei_v;
			if (offset_nei_u >= off_u_end || offset_nei_v >= off_v_end) {
				return cn_count;
			}
		}
	}
}

template<typename T>
__device__ int ComputeCNGallopingSingleDirDevice(uint32_t *node_off, /*card: |V|+1*/
T *edge_dst, int u, int v) {
	auto cn_count = 0;
	if (node_off[u + 1] - node_off[u] > node_off[v + 1] - node_off[v]) {
		auto tmp = u;
		u = v;
		v = tmp;
	}
	auto offset_nei_u = node_off[u], offset_nei_v = node_off[v];
	auto off_u_end = node_off[u + 1], off_v_end = node_off[v + 1];

	while (true) {
		while (edge_dst[offset_nei_u] < edge_dst[offset_nei_v]) {
			++offset_nei_u;
			if (offset_nei_u >= off_u_end) {
				return cn_count;
			}
		}

		offset_nei_v = GallopingSearchDevice(edge_dst, offset_nei_v, off_v_end, edge_dst[offset_nei_u]);
		if (offset_nei_v >= off_v_end) {
			return cn_count;
		}

		if (edge_dst[offset_nei_u] == edge_dst[offset_nei_v]) {
			cn_count++;
			++offset_nei_u;
			++offset_nei_v;
			if (offset_nei_u >= off_u_end || offset_nei_v >= off_v_end) {
				return cn_count;
			}
		}
	}
}

template<typename T>
__device__ int ComputeCNGallopingSingleDirDeviceFixed(uint32_t *node_off, /*card: |V|+1*/
T *edge_dst, uint32_t offset_nei_u, uint32_t offset_nei_v, uint32_t off_u_end, uint32_t off_v_end) {
	auto cn_count = 0;
	while (true) {
		while (edge_dst[offset_nei_u] < edge_dst[offset_nei_v]) {
			++offset_nei_u;
			if (offset_nei_u >= off_u_end) {
				return cn_count;
			}
		}

		offset_nei_v = GallopingSearchDevice(edge_dst, offset_nei_v, off_v_end, edge_dst[offset_nei_u]);
		if (offset_nei_v >= off_v_end) {
			return cn_count;
		}

		if (edge_dst[offset_nei_u] == edge_dst[offset_nei_v]) {
			cn_count++;
			++offset_nei_u;
			++offset_nei_v;
			if (offset_nei_u >= off_u_end || offset_nei_v >= off_v_end) {
				return cn_count;
			}
		}
	}
}

template<typename T>
__device__ int ComputeCNHybridDevice(uint32_t *node_off, /*card: |V|+1*/
T *edge_dst, int u, int v) {
	int g_deg_u = node_off[u + 1] - node_off[u];
	int g_deg_v = node_off[v + 1] - node_off[v];
	if (g_deg_u / 50 > g_deg_v || g_deg_v / 50 > g_deg_u) {
		return ComputeCNGallopingSingleDirDevice(node_off, edge_dst, u, v);
	} else {
		return ComputeCNNaiveStdMergeDevice(node_off, edge_dst, u, v);
	}
}
