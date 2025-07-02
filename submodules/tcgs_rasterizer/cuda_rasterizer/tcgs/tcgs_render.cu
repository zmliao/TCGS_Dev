// Copyright (c) 2025 TCGS GROUP. MIT License. See LICENSE for details.
#include "tcgs.h"
#include "tcgs_utils.h"
#include <cuda.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include "cuda_runtime.h"


namespace cg = cooperative_groups;


using namespace TCGS_UTIL;

__device__ void identify_pixel_properties(
    cg::thread_block &block, int thread_id, int warp_id,
    int width, int height,
    bool &inside, int &tile_id, int &pix_id,
    float2 &pixf_mid, float2 &pixf_local
)
{
    //identify tile ranges
    uint horizontal_blocks = (width + BLOCK_X_TCGS - 1) / BLOCK_X_TCGS;
    uint2 pix_min = make_uint2(block.group_index().x * BLOCK_X_TCGS, block.group_index().y * BLOCK_Y_TCGS);
    uint2 pix_max = make_uint2(min(pix_min.x + BLOCK_X_TCGS, width), min(pix_min.y + BLOCK_Y_TCGS , height));

    //identify global coordinates, each warp contains 8x4 pixels
    uint2 pix = make_uint2(
        pix_min.x + (((warp_id >> 2) << 3) | (thread_id & 7)), 
        pix_min.y + ((thread_id & 127) >> 3));
    
    //identify whether the pixel is inside the image
    inside = pix.x < width && pix.y < height;

    //identify pixel and tile indices
    tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
    pix_id = width * pix.y + pix.x;

    //identify local coordinates
    pixf_mid = make_float2(
        __uint2float_rn(pix_min.x) + 7.5f, 
        __uint2float_rn(pix_min.y) + 7.5f
    );
    pixf_local = make_float2(
        __uint2float_rn(pix.x) - pixf_mid.x,
        __uint2float_rn(pix.y) - pixf_mid.y
    );
}


//convert pixel at local coordinates into pixel vectors
__forceinline__ __device__ uint4 pix2vec(
    float2 pixf_local
)
{
    uint4 pixel_vector;
    pixel_vector.x = float22reg(pixf_local.x, pixf_local.y);
    pixel_vector.y = float22reg(pixf_local.x * pixf_local.x, pixf_local.y * pixf_local.y);
    pixel_vector.z = float22reg(1.0f / 3.0f, pixf_local.x * pixf_local.y);
    pixel_vector.w = float22reg(1.0f / 3.0f, 1.0f / 3.0f);
    return pixel_vector;
}

//convert Gaussian primitives into Gaussian vectors
__forceinline__ __device__ uint4 gs2vec(
    float4 conic_opacity,
    float2 means,
    float2 pixf_mid
)
{
    //Converse global coordinates into local coordinates
    means.x = pixf_mid.x - means.x;
    means.y = pixf_mid.y - means.y;

    //compute the constant term of Gaussian vector
    float constant = 
        conic_opacity.w + 
        conic_opacity.x * means.x * means.x + 
        conic_opacity.y * means.x * means.y +
        conic_opacity.z * means.y * means.y;
    
    //compute the gaussian vector
    uint4 gaussian_vector;
    gaussian_vector.x = float22reg(
        2.0f * conic_opacity.x * means.x + conic_opacity.y * means.y,
        2.0f * conic_opacity.z * means.y + conic_opacity.y * means.x
    );
    gaussian_vector.y = float22reg(conic_opacity.x, conic_opacity.z);
    gaussian_vector.z = float22reg(constant, conic_opacity.y);
    gaussian_vector.w = float22reg(constant, constant);
    return gaussian_vector;
}

//Stack vectors into matrices
__forceinline__ __device__ void vec2mat(
    uint4 vecs,
    uint* mat_smem,
    uint thread_id
)
{
    uint addr = (thread_id << 2);
    mat_smem[addr] = vecs.x;
    mat_smem[addr | 1] = vecs.y;
    mat_smem[addr | 2] = vecs.z;
    mat_smem[addr | 3] = vecs.w;
}

//Store exponents registers into matrix
__forceinline__ __device__ void store_exponent_mat(
    uint reg0, uint reg1, uint reg2, uint reg3,
    uint* mat_smem
)
{
    mat_smem[0] = reg0;
    mat_smem[8] = reg1;
    mat_smem[16] = reg2;
    mat_smem[24] = reg3;
}

//Culling and Alpha-blending
__forceinline__ __device__ uint2 culling_and_blending(
    uint* exponent_matrix, // Shared Memory
    uint2* channels_smem,
    half &T, int gs_index, int thread_id, uint2 RGBD
)
{
#pragma unroll
    for(int k = 0; k < 8; ++k)
    {
        half2 exponents = *reinterpret_cast<half2*>(exponent_matrix + ((k<<8)|thread_id));
        //beta > -ln(225) and beta < 0, otherwise culled
        if(__hgt(exponents.x, __float2half_rn(-7.995f)) && __hlt(exponents.x, __float2half_rn(0.0f)))
        {
            //alpha-blending
            half alpha = __hmul(T, __hmin(__float2half_rn(0.99f), fast_ex2_f16(exponents.x)));
            T = __hsub(T, alpha);
            uint2 channel = channels_smem[gs_index|(k<<1)];
            uint alpha2 = half22uint(make_half2(alpha, alpha));
            RGBD.x = fast_fma_rn_ftz_f16x2(channel.x, alpha2, RGBD.x);
            RGBD.y = fast_fma_rn_ftz_f16x2(channel.y, alpha2, RGBD.y);
        }

        if(__hgt(exponents.y, __float2half_rn(-7.995f)) && __hlt(exponents.y, __float2half_rn(0.0f)))
        {
            //alpha-blending
            half alpha = __hmul(T, __hmin(__float2half_rn(0.99f), fast_ex2_f16(exponents.y)));
            T = __hsub(T, alpha);
            uint2 channel = channels_smem[gs_index|(k<<1)|1];
            uint alpha2 = half22uint(make_half2(alpha, alpha));
            RGBD.x = fast_fma_rn_ftz_f16x2(channel.x, alpha2, RGBD.x);
            RGBD.y = fast_fma_rn_ftz_f16x2(channel.y, alpha2, RGBD.y);
        }
    }
    return RGBD;
}

//Main Kernel of TC-GS
__global__ void renderCUDA_TCGS(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int width, int height,
    const float2* __restrict__ points_xy_image,
    const uint2* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    float* __restrict__ invdepth,
    bool is_taming = false,
    const uint32_t* __restrict__ per_tile_bucket_offset = nullptr,
    uint32_t* __restrict__ bucket_to_tile = nullptr,
    half* __restrict__ sampled_T = nullptr,
    uint2* __restrict__ sampled_ar = nullptr,
    uint32_t* __restrict__ max_contrib = nullptr
) 
{
    auto block = cg::this_thread_block();
    int thread_id = block.thread_rank();
    int warp_id = thread_id / WAPR_SIZE;

    bool inside; //Whether the pixel is insice the image
    int pix_id;  //Pixel index of the screen space
    int tile_id; //Tile index of the screen space
    float2 pixf_mid; //Center pixel inside the tile
    float2 pixf_local; //Local coordinates of the pixel

    identify_pixel_properties(
        block, thread_id, warp_id,
        width, height,
        inside, tile_id, pix_id, pixf_mid, pixf_local
    );

    //Define Matrices
    __shared__ uint multiuse_matrix[BLOCK_SIZE_TCGS * REDUCE_SIZE / 2]; //multiple use for storing 
    __shared__ uint exponent_matrix[BLOCK_SIZE_TCGS * VECTOR_SIZE];
    __shared__ uint2 channels_smem[BLOCK_SIZE_TCGS];
    uint* exponent_matrix_addr = exponent_matrix + ((thread_id&3) << 8) + (warp_id<<5) + ((thread_id&31) >> 2);
    uint pixmat_reg[4];

    //load pixel_vector
    uint4 pix_vec = pix2vec(pixf_local);
    vec2mat(pix_vec, multiuse_matrix, thread_id);
    __syncwarp();
    load_matrix_x4(
        pixmat_reg[0], pixmat_reg[1], pixmat_reg[2], pixmat_reg[3],
        multiuse_matrix + (thread_id<<2)
    );

    bool done = !inside;
    bool warp_done = (__ballot_sync(~0, done) == (~0));
    uint2 range = ranges[tile_id];
    int toDo = range.y - range.x;
    const int rounds = (toDo + BLOCK_SIZE_TCGS - 1) / BLOCK_SIZE_TCGS;

    half T = __float2half_rn(1.0f);
    uint num_contrib = 0u;
    uint2 RGBD = make_uint2(0u, 0u);

    uint32_t bbm = 0;
    if(is_taming){
        bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
        int num_buckets = (toDo + 31) / 32;
        for (int i = 0; i < (num_buckets + BLOCK_SIZE_TCGS - 1) / BLOCK_SIZE_TCGS; ++i) {
            int bucket_idx = i * BLOCK_SIZE_TCGS + block.thread_rank();
            if (bucket_idx < num_buckets) {
                bucket_to_tile[bbm + bucket_idx] = tile_id;
            }
        }
    }
    for(int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE_TCGS)
    {
        int num_done = __syncthreads_count(done);
        if(num_done == BLOCK_SIZE_TCGS)
            break;

        int progress = ((i<<8) | thread_id);
        uint4 gaussian_vec = make_uint4(0u, 0u, 0u, float22reg(1.0, 1.0));
        uint2 channel = make_uint2(0u, 0u);
        
        //load channels ans gaussian vectors
        if(range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            float4 conics = conic_opacity[coll_id];
            float2 means = points_xy_image[coll_id];
            gaussian_vec = gs2vec(conics, means, pixf_mid);
            channel = features[coll_id];
        }
        vec2mat(gaussian_vec, multiuse_matrix, thread_id);
        channels_smem[thread_id] = channel;

        block.sync();
        const int gs_num = min(BLOCK_SIZE_TCGS, toDo);
        for(int j = 0; !warp_done && j < gs_num; j += REDUCE_SIZE){
            if (is_taming && j % 32 == 0) {
				sampled_T[(bbm * BLOCK_SIZE_TCGS) + block.thread_rank()] = T;
				sampled_ar[(bbm * BLOCK_SIZE_TCGS) + block.thread_rank()] = RGBD;
				++bbm;
			}
            //1. load the Gaussian matrix V
            uint gsmat_reg[2];
            load_matrix_x2(gsmat_reg[0], gsmat_reg[1], multiuse_matrix + (j<<2) + ((thread_id&31)<<2));

            //2. compute exponents matrix B = U^T * V = [U1, U2]^T * [V1, V2]
            //The division strategy is to align the format of instructions in ptx
            //2.1 Compute B_1 = [U1^T * V1, U2^T * V1]
            uint expmat_reg[4] = {0u, 0u, 0u, 0u};
            mma_16x8x8_f16_f16(expmat_reg[0], expmat_reg[1],
                pixmat_reg[0], pixmat_reg[1], gsmat_reg[0], expmat_reg[0], expmat_reg[1]);
            mma_16x8x8_f16_f16(expmat_reg[2], expmat_reg[3],
                pixmat_reg[2], pixmat_reg[3], gsmat_reg[0], expmat_reg[2], expmat_reg[3]);
            store_exponent_mat(expmat_reg[0], expmat_reg[1], expmat_reg[2], expmat_reg[3], exponent_matrix_addr);
            __syncwarp();
            
            //2.2 Compute B_2 = [U1^T * V2, U2^T * V2]
            expmat_reg[0] = expmat_reg[1] = expmat_reg[2] = expmat_reg[3] = 0u;
            mma_16x8x8_f16_f16(expmat_reg[0], expmat_reg[1],
                pixmat_reg[0], pixmat_reg[1], gsmat_reg[1], expmat_reg[0], expmat_reg[1]);
            mma_16x8x8_f16_f16(expmat_reg[2], expmat_reg[3],
                pixmat_reg[2], pixmat_reg[3], gsmat_reg[1], expmat_reg[2], expmat_reg[3]);
            store_exponent_mat(expmat_reg[0], expmat_reg[1], expmat_reg[2], expmat_reg[3], exponent_matrix_addr + 1024);
            __syncwarp();

            //3. culling and alpha-blending
            RGBD = culling_and_blending(exponent_matrix, channels_smem, T, j, thread_id, RGBD);
            if(__hlt(T, __float2half_rn(0.0001f)))
                done = true;
            if(__ballot_sync(~0, done) == (~0))
                warp_done = true;
            num_contrib += REDUCE_SIZE;
        }
    }
    num_contrib = min(num_contrib, range.y - range.x);
    //output colors and other informations
    if(inside)
    {
        float Tf = __half2float(T);
        half2 RGh = uint2half2(RGBD.x);
        half2 BDh = uint2half2(RGBD.y);
        final_T[pix_id] = Tf;

        out_color[pix_id] = __half2float(RGh.x) + Tf * bg_color[0];
        out_color[pix_id + width * height] = __half2float(RGh.y) + Tf * bg_color[1];
        out_color[pix_id + 2 * width * height] = __half2float(BDh.x) + Tf * bg_color[2];
        if(invdepth)
            invdepth[pix_id] = __half2float(BDh.y);
        
        n_contrib[pix_id] = num_contrib;
    }
    if(is_taming)
    {
        typedef cub::BlockReduce<uint32_t, BLOCK_SIZE_TCGS> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        num_contrib = BlockReduce(temp_storage).Reduce(num_contrib, cub::Max());
        if (block.thread_rank() == 0) {
            max_contrib[tile_id] = num_contrib;
        }
    }
}


__global__ void transform_coefs(
    const int P,
    const float* colors,
    const float* depths,
    float4* conic_opacity,
    uint2* feature_encoded,
    float* invdepth
)
{
    auto idx = cg::this_grid().thread_rank();
    if(idx >= P)
        return;
    
    //Preprocess the inverse covariance for vectorizing 
    float4 conics = conic_opacity[idx];
    conic_opacity[idx] = make_float4(
        LOG2E_N_2 * conics.x, LOG2E_N * conics.y, LOG2E_N_2 * conics.z, fast_lg2_f32(conics.w)
    );

    //Compress the colors into fp16
    float4 features = make_float4(
        colors[idx * 3], colors[idx * 3 + 1], colors[idx * 3 + 2], 0.0f
    );
    if(invdepth)
        features.w = 1.0f / depths[idx];
    uint RG = float22reg(features.x, features.y);
    uint BD = float22reg(features.z, features.w);
    feature_encoded[idx] = make_uint2(RG, BD);
}


void TCGS::renderCUDA_Forward(
    const dim3 grid,
    const dim3 block,
    const uint2* ranges,
    const uint* point_list,
    int width,
    int height,
    int P,
    const float2* means2D,
    const float* colors,
    float4* conic_opacity,
    float* final_T,
    uint* n_contrib,
    const float* bg_color,
    float* out_color,
    float* depths,
    float* depth
)
{
    //Preprocess for TCGS
    uint2* feature_encoded = nullptr;
    cudaMalloc(&feature_encoded, P * sizeof(uint2));
    transform_coefs<< <(P + 255) / 256, 256>> >(
        P, colors, depths, conic_opacity, feature_encoded, depth
    );

    //Running TCGS
    renderCUDA_TCGS<< <grid, block>> >(
        ranges, point_list,
        width, height,
        means2D, feature_encoded, conic_opacity,
        final_T, n_contrib,
        bg_color, out_color, depth
    );
    //
    cudaFree(feature_encoded);
    
}

CudaRasterizer_TCGS::SampleState CudaRasterizer_TCGS::SampleState::fromChunk(char* &chunk, size_t B, size_t P)
{
    SampleState sample;
    obtain(chunk, sample.bucket_to_tile, B * BLOCK_SIZE_TCGS, 128);
    obtain(chunk, sample.T, B * BLOCK_SIZE_TCGS, 128);
    obtain(chunk, sample.ar, B * BLOCK_SIZE_TCGS, 128);
    obtain(chunk, sample.channels, P, 128);
    return sample;
}

void TCGS::renderCUDA_Forward_Taming(
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
)
{
    auto sample = CudaRasterizer_TCGS::SampleState::fromChunk(sample_chunkptr, B, P);

    //Preprocess for TCGS
    // uint2* feature_encoded = nullptr;
    // cudaMalloc(&feature_encoded, P * sizeof(uint2));
    transform_coefs<< <(P + 255) / 256, 256>> >(
        P, features, depths, conic_opacity, sample.channels, depth
    );

    //Running TCGS
    renderCUDA_TCGS<< <grid, block>> >(
        ranges, point_list,
        width, height,
        means2D, sample.channels, conic_opacity,
        final_T, n_contrib,
        bg_color, out_color, depth,
        true, bucket_offsets,
        sample.bucket_to_tile, sample.T, sample.ar, max_contrib
    );

}

__global__ void renderCUDA_Backward_Taming_TCGS(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int width, int height, int B,
    const uint32_t* __restrict__ per_tile_bucket_offset,
    const uint32_t* __restrict__ bucket_to_tile,
    const half* __restrict__ sampled_T,
    const uint2* __restrict__ sampled_ar,
    const float* __restrict__ bg_color,
    const float2* __restrict__ points_xy_image,
    const float4* __restrict__ conic_opacity,
    const uint2* __restrict__ channels,
    const float* __restrict__ final_Ts,
    const uint32_t* __restrict__ n_contrib,
    const uint32_t* __restrict__ max_contrib,
    const float* __restrict__ pixel_colors,
    const float* __restrict__ pixel_invDepths,
    const float* __restrict__ dL_dpixels,
    const float* __restrict__ dL_invdepths,
    float3* __restrict__ dL_dmean2D,
    float4* __restrict__ dL_dconic2D,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dcolors,
    float* __restrict__ dL_dinvdepths
){
	// global_bucket_idx = warp_idx
	auto block = cg::this_thread_block();
	auto my_warp = cg::tiled_partition<32>(block);
	uint32_t global_bucket_idx = block.group_index().x * my_warp.meta_group_size() + my_warp.meta_group_rank();
	bool valid_bucket = global_bucket_idx < (uint32_t) B;
	if (!valid_bucket) return;

	bool valid_splat = false;

	uint32_t tile_id, bbm;
	uint2 range;
	int num_splats_in_tile, bucket_idx_in_tile;
	int splat_idx_in_tile, splat_idx_global;

	tile_id = bucket_to_tile[global_bucket_idx];
	range = ranges[tile_id];
	num_splats_in_tile = range.y - range.x;
	// What is the number of buckets before me? what is my offset?
	bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	bucket_idx_in_tile = global_bucket_idx - bbm;
	splat_idx_in_tile = bucket_idx_in_tile * 32 + my_warp.thread_rank();
	splat_idx_global = range.x + splat_idx_in_tile;
	valid_splat = (splat_idx_in_tile < num_splats_in_tile);

	// if first gaussian in bucket is useless, then others are also useless
	if (bucket_idx_in_tile * 32 >= max_contrib[tile_id]) {
		return;
	}

	// Load Gaussian properties into registers
	int gaussian_idx = 0;
	float2 xy = {0.0f, 0.0f};
	float4 con_o = {0.0f, 0.0f, 0.0f, 0.0f};
	float4 RGBD = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float opacity = 0.0f;

	if (valid_splat) {
		gaussian_idx = point_list[splat_idx_global];
		xy = points_xy_image[gaussian_idx];
		con_o = conic_opacity[gaussian_idx];
        opacity = fast_ex2_f32(con_o.w);
        RGBD = reg22float4(channels[gaussian_idx]); 
	}

	// Gradient accumulation variables
	float Register_dL_dmean2D_x = 0.0f;
	float Register_dL_dmean2D_y = 0.0f;
	float Register_dL_dconic2D_x = 0.0f;
	float Register_dL_dconic2D_y = 0.0f;
	float Register_dL_dconic2D_w = 0.0f;
	float Register_dL_dopacity = 0.0f;
	float3 Register_dL_dcolors = {0.0f, 0.0f, 0.0f};
	float Register_dL_dinvdepths = 0.0f;
	
	// tile metadata
	const uint32_t horizontal_blocks = (width + BLOCK_X_TCGS - 1) / BLOCK_X_TCGS;
	const uint2 tile = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
	const uint2 pix_min = {tile.x * BLOCK_X_TCGS, tile.y * BLOCK_Y_TCGS};

	// values useful for gradient calculation
	float T;
	// float T_final;
    float dL_dBG;
	float last_contributor;
	float4 ar;
	float4 dL_dchannel;

	// iterate over all pixels in the tile
	for (int i = 0; i < BLOCK_SIZE_TCGS + 31; ++i) {
		// SHUFFLING

		// At this point, T already has my (1 - alpha) multiplied.
		// So pass this ready-made T value to next thread.
		T = my_warp.shfl_up(T, 1);
		last_contributor = my_warp.shfl_up(last_contributor, 1);
		dL_dBG = my_warp.shfl_up(dL_dBG, 1);
        ar = my_warp.shfl_up(ar, 1);
        dL_dchannel = my_warp.shfl_up(dL_dchannel, 1);

		// which pixel index should this thread deal with?
		int idx = i - my_warp.thread_rank();
        if(idx < 0)
            continue;
		const uint2 pix = make_uint2(pix_min.x + (idx & 7) + ((idx >> 7) <<3), pix_min.y + ((idx & 127) >> 3));
		const uint32_t pix_id = width * pix.y + pix.x;
		const float2 pixf = {(float) pix.x, (float) pix.y};
		bool valid_pixel = pix.x < width && pix.y < height;
		
		// every 32nd thread should read the stored state from memory
		// TODO: perhaps store these things in shared memory?
		if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < BLOCK_SIZE_TCGS) {

			T = __half2float(sampled_T[global_bucket_idx * BLOCK_SIZE_TCGS + idx]);
            ar = reg22float4(sampled_ar[global_bucket_idx * BLOCK_SIZE_TCGS + idx]);

            ar.x = ar.x - pixel_colors[pix_id];
            ar.y = ar.y - pixel_colors[height * width + pix_id];
            ar.z = ar.z - pixel_colors[2 * height * width + pix_id];
            ar.w = ar.w - (pixel_invDepths ? pixel_invDepths[pix_id] : 0.0f);
        
			float T_final = final_Ts[pix_id];
			last_contributor = n_contrib[pix_id];
            dL_dchannel.x = dL_dpixels[pix_id];
            dL_dchannel.y = dL_dpixels[height * width + pix_id];
            dL_dchannel.z = dL_dpixels[2 * height * width + pix_id];
			dL_dchannel.w = dL_invdepths[pix_id];
            dL_dBG = bg_color[0] * dL_dchannel.x + bg_color[1] * dL_dchannel.y + bg_color[2] * dL_dchannel.z;
            dL_dBG *= (-T_final);
		}

		// do work
		if (valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE_TCGS) {
			if (width <= pix.x || height <= pix.y) 
                continue;

			if (splat_idx_in_tile >= last_contributor) 
                continue;

			// compute blending values
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float power = con_o.x * d.x * d.x + con_o.y * d.x * d.y + con_o.z * d.y * d.y + con_o.w;
			if (power > 0.0f) 
                continue;
                
			const float alpha = min(0.99f, fast_ex2_f32(power));
			if (alpha < 1.0f / 255.0f)
                continue;
			const float weight = alpha * T;
            T = T - weight;
			// add the gradient contribution of this pixel's colour to the gaussian
            ar.x += weight * RGBD.x;
            ar.y += weight * RGBD.y;
            ar.z += weight * RGBD.z;
            ar.w += weight * RGBD.w;

            Register_dL_dcolors.x += weight * dL_dchannel.x;
            Register_dL_dcolors.y += weight * dL_dchannel.y;
            Register_dL_dcolors.z += weight * dL_dchannel.z;
            Register_dL_dinvdepths += weight * dL_dchannel.w;

            const float coef = alpha / (1.0f - alpha);
            float dL_dexpo = 
                (RGBD.x * weight + coef * ar.x) * dL_dchannel.x +
                (RGBD.y * weight + coef * ar.y) * dL_dchannel.y +
                (RGBD.z * weight + coef * ar.z) * dL_dchannel.z +
                (RGBD.w * weight + coef * ar.w) * dL_dchannel.w;
            
            dL_dexpo += coef * dL_dBG;

            Register_dL_dopacity += dL_dexpo / opacity;
            Register_dL_dconic2D_x += dL_dexpo * d.x * d.x;
            Register_dL_dconic2D_y += dL_dexpo * d.x * d.y;
            Register_dL_dconic2D_w += dL_dexpo * d.y * d.y;
			Register_dL_dmean2D_x += dL_dexpo * (2.0f * con_o.x * d.x + con_o.y * d.y);
            Register_dL_dmean2D_y += dL_dexpo * (2.0f * con_o.z * d.y + con_o.y * d.x);
		}
	}

	// finally add the gradients using atomics
	if (valid_splat) {
		atomicAdd(&dL_dmean2D[gaussian_idx].x, LN2_2 * width * Register_dL_dmean2D_x);
		atomicAdd(&dL_dmean2D[gaussian_idx].y, LN2_2 * height * Register_dL_dmean2D_y);
		atomicAdd(&dL_dconic2D[gaussian_idx].x, -0.5f * Register_dL_dconic2D_x);
		atomicAdd(&dL_dconic2D[gaussian_idx].y, -0.5f * Register_dL_dconic2D_y);
		atomicAdd(&dL_dconic2D[gaussian_idx].w, -0.5f * Register_dL_dconic2D_w);
		atomicAdd(&dL_dopacity[gaussian_idx], Register_dL_dopacity);
		atomicAdd(&dL_dcolors[gaussian_idx * 3], Register_dL_dcolors.x);
        atomicAdd(&dL_dcolors[gaussian_idx * 3 + 1], Register_dL_dcolors.y);
        atomicAdd(&dL_dcolors[gaussian_idx * 3 + 2], Register_dL_dcolors.z);
		atomicAdd(&dL_dinvdepths[gaussian_idx], Register_dL_dinvdepths);
	}
}


void TCGS::renderCUDA_Backward_Taming(
    const dim3 grid,
    const dim3 block,
    const uint2* ranges,
    const uint* point_list,
    int width, int height, int R, int B, int P,
    const uint* bucket_offsets,
    char* &sample_chunkptr,
    const float* bg_color,
    const float2* means2D,
    const float4* conic_opacity,
    const float* colors,
    const float* depths,
    const float* final_Ts,
    const uint32_t* n_contrib,
    const uint32_t* max_contrib,
    const float* pixel_colors,
    const float* pixel_invDepths,
    const float* dL_dpixels,
    const float* dL_invdepths,
    float3* dL_dmean2D,
    float4* dL_dconic2D,
    float* dL_dopacity,
    float* dL_dcolors,
    float* dL_dinvdepths
){
    auto sample = CudaRasterizer_TCGS::SampleState::fromChunk(sample_chunkptr, B, P);
    
    const int THREADS = 32;

    renderCUDA_Backward_Taming_TCGS<< <((B * 32) + THREADS - 1) / THREADS, THREADS>> >(
        ranges, point_list,
        width, height, B,
        bucket_offsets, sample.bucket_to_tile,
        sample.T, sample.ar,
        bg_color, means2D, conic_opacity,
        sample.channels, final_Ts, n_contrib, max_contrib,
        pixel_colors, pixel_invDepths,
        dL_dpixels, dL_invdepths,
        dL_dmean2D, dL_dconic2D,
        dL_dopacity, dL_dcolors, dL_dinvdepths
    );
}