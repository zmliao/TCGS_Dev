// Copyright (c) 2025 TCGS GROUP. MIT License. See LICENSE for details.
#ifndef TCGS_H
#define TCGS_H

#ifdef _WIN32
    #define TCGS_API __declspec(dllexport)
#else
    #define TCGS_API __attribute__((visibility("default")))
#endif

#include<cuda_fp16.h>

namespace CudaRasterizer_TCGS
{
    template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

    struct SampleState
	{
		uint32_t *bucket_to_tile;
		half *T;
        uint2 *ar;
		static SampleState fromChunk(char*& chunk, size_t B);
	};

    template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};


namespace TCGS
{
    void renderCUDA_Forward(
        const dim3 grid, 
        const dim3 block,
        const uint2* ranges,
        const uint* point_list,
        int width,
        int height,
        int P,
        const float2* means2D,
        const float* features,
        float4* conic_opacity,
        float* final_T,
        uint* n_contrib,
        const float* bg_color,
        float* out_color,
        float* depths,
        float* depth
    );

    void renderCUDA_Forward_Taming(
        const dim3 grid, 
        const dim3 block,
        const uint2* ranges,
        const uint* point_list,
        const uint* bucket_offsets,
        char* &sample_chunkptr,
        int width,
        int height,
        int P, int B, // B is the number of buckets
        const float2* means2D,
        const float* features,
        float4* conic_opacity,
        float* final_T,
        uint* n_contrib,
        uint* max_contrib,
        const float* bg_color,
        float* out_color,
        float* depths,
        float* depth
    );
};

#endif 