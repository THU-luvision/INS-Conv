// Added by Tian Zheng
// t-zheng@outlook.com
// cuda kernels and launchers
#include "kernel_hash.cuh"
#include "../Metadata/Metadata.h"
#include <assert.h>

__device__ __constant__ Int rot_index[27 * 6] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
    24,25,26,21,22,23,18,19,20,15,16,17,12,13,14,9,10,11,6,7,8,3,4,5,0,1,2,
    6,7,8,15,16,17,24,25,26,3,4,5,12,13,14,21,22,23,0,1,2,9,10,11,18,19,20,
    18,19,20,9,10,11,0,1,2,21,22,23,12,13,14,3,4,5,24,25,26,15,16,17,6,7,8,
    2,11,20,5,14,23,8,17,26,1,10,19,4,13,22,7,16,25,0,9,18,3,12,21,6,15,24,
    18,9,0,21,12,3,24,15,6,19,10,1,22,13,4,25,16,7,20,11,2,23,14,5,26,17,8
    };

__device__ __constant__ Int reverse_index[27] = {
    26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
};

/* structure for 3d point */
typedef struct point_3d {
    Int x;
    Int y;
    Int z;

    __device__ uint32_t hash() const  {
        Int h = 16777619;
        h *= 2166136261;
        h ^= x;
        h *= 2166136261;
        h ^= y;
        h *= 2166136261;
        h ^= z;
        return (uint32_t)h;
    }

    __device__ bool operator== (const point_3d& other)   {
        return (x == other.x) && (y == other.y) && (z == other.z);
    }
}point_t;

typedef RH_hash_table<point_t, Int> hashtable_t;

__global__ void _d_GetPointsWithinInputRegion_3d(const Int* d_points, 
                                    Int* d_output_points, Int num)    {
    /*  for x in [lb.x ... ub.x] :
    *       for y ...
    *           for z ...
    */   
    Int out_location[3];
    uint32_t hash;
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < num)  {
        //Get input location
#pragma unroll
        for(Int i = 0; i < 3; i++)  {
            out_location[i] = d_points[tid + num * i];
        }
        // write into output array
        Int index = 0;
#pragma unroll
        for(Int i = -1; i < 2; i++)    {
#pragma unroll
            for(Int j = -1; j < 2; j++)    {
#pragma unroll
                for(Int k = -1; k < 2; k++)    {
                    hash = ((out_location[2]+k) << 21 | (out_location[1]+j) << 10 | (out_location[0]+i));
                    hash &= 0x7FFFFFFF;
                    d_output_points[tid * 27 + index] = hash;
                    index++;
                }
            }
        }
        tid += blockDim.x * gridDim.x;
    }
}

void dGetPointsWithinInputRegion (const Int* d_points, Int* d_output_points, Int num, Int size, Int Dimension) {
    if(size != 3)   {
        fprintf(stderr, "size != 3\n");
        abort();
    }
    if(Dimension != 3)  {
        fprintf(stderr, "dimension != 3\n");
        abort();
    }
    _d_GetPointsWithinInputRegion_3d<<<512,512>>>(d_points, d_output_points, num);
}

__global__ void _dGenerateRulebook (Int* d_global_query_result,    // size = NActivePoints * volume
                                    Int* d_out_rules,              // size = volume * (NActivePoints * 2)
                                    unsigned* d_out_rules_flag,    // size = volume * (NActivePoints * 2)
                                    Int NActivePoints,
                                    Int volume,
                                    Int ctr)
{
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Int current_result;
    Int current_address;
    while(tid < NActivePoints)  {
        current_address = tid;
        for(Int i = 0; i < volume; i++) {
            current_result = d_global_query_result[tid * volume + i];
            if(current_result != -1) {
                d_out_rules[(NActivePoints * 2) * i + tid * 2] = current_result + ctr;
                d_out_rules[(NActivePoints * 2) * i + tid * 2 + 1] = current_address + ctr;
                d_out_rules_flag[(NActivePoints * 2) * i + tid * 2] = 1;
                d_out_rules_flag[(NActivePoints * 2) * i + tid * 2 + 1] = 1;
            }
        }
        tid += blockDim.x * gridDim.x;
    }
}

void dGenerateRulebook (Int* d_global_query_result,    // size = NActivePoints * volume
                        Int* d_out_rules,              // size = volume * (NActivePoints * 2)
                        unsigned* d_out_rules_flag,    // size = volume * (NActivePoints * 2)
                        Int NActivePoints,
                        Int volume,
                        Int ctr)
{
    _dGenerateRulebook<<<512,512>>> (d_global_query_result,    // size = NActivePoints * volume
                                     d_out_rules,              // size = volume * (NActivePoints * 2)
                                     d_out_rules_flag,    // size = volume * (NActivePoints * 2)
                                     NActivePoints,
                                     volume,
                                     ctr);
}


__global__ void _dGenerateIncreRulebook (Int* d_global_query_result,    // size = NActivePoints * volume
                                    Int* d_output_query_result,
                                    Int* d_out_rules,              // size = volume * (NActivePoints * 2)
                                    unsigned* d_out_rules_flag,    // size = volume * (NActivePoints * 2)
                                    Int NActivePoints,
                                    Int volume,
                                    Int ctr)
{
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Int current_result;
    Int current_address;
    while(tid < NActivePoints)  {
        current_address = d_output_query_result[tid];
        for(Int i = 0; i < volume; i++) {
            current_result = d_global_query_result[tid * volume + i];
            if(current_result != -1) {
                d_out_rules[(NActivePoints * 2) * i + tid * 2] = current_result + ctr;
                d_out_rules[(NActivePoints * 2) * i + tid * 2 + 1] = current_address + ctr;
                d_out_rules_flag[(NActivePoints * 2) * i + tid * 2] = 1;
                d_out_rules_flag[(NActivePoints * 2) * i + tid * 2 + 1] = 1;
            }
        }
        tid += blockDim.x * gridDim.x;
    }
}

void dGenerateIncreRulebook (Int* d_global_query_result,    // size = NActivePoints * volume
                        Int *d_output_query_result,
                        Int* d_out_rules,              // size = volume * (NActivePoints * 2)
                        unsigned* d_out_rules_flag,    // size = volume * (NActivePoints * 2)
                        Int NActivePoints,
                        Int volume,
                        Int ctr)
{
    _dGenerateIncreRulebook<<<512,512>>> (d_global_query_result,    // size = NActivePoints * volume
                                    d_output_query_result,
                                     d_out_rules,              // size = volume * (NActivePoints * 2)
                                     d_out_rules_flag,    // size = volume * (NActivePoints * 2)
                                     NActivePoints,
                                     volume,
                                     ctr);
}



__global__ void _d_GetBlockGrid(Int *d_input_point, Int *d_blockid, Int block_size, Int num) {
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < num)  {
        d_blockid[tid] = d_input_point[tid] / block_size;
        tid += blockDim.x * gridDim.x;
    }
}

void dGetBlockGrid(Int *d_input_point, Int *d_blockid, Int block_size, Int length) {
    _d_GetBlockGrid<<<512,512>>> (d_input_point, d_blockid, block_size, length);
}



__global__ void _dGetAddressbookSize(
    hashtable_t *d_hashtables,                   // size = [num_block, ]
    Int* d_output_input_address_size,                // size = [num_block, ]
    Int num_block)
{
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < num_block)  {
        d_output_input_address_size[tid] = d_hashtables[tid].num_elems;
        tid += blockDim.x * gridDim.x;
    }
}


/*
 *  Changelog: 
 *  1. Removed conv_structure
 *  2. replace max_point_per_block with MAX_INPUT_ADDRESS
 * 
 */
__global__ void _dGenerateChuckRulebook(Int* d_input_point_query_result,             // size = num_active_point * 27
                                        Int* d_ori_index,                            // size = num_active_point
                                        Int* d_active_output_point_blockid,          // size = num_active_point
                                        Int* d_block_active_output_count,            /* initialize with 0 */

                                        Int* d_block_input_address_list,
                                        Int* d_block_input_address_list_size,

                                        short* d_output_rulebook,                      // dim = (2, num_block, max_input_address * 27) 
                                        Int* d_output_output_address,                // size = [num_block, max_input_address]

                                        Int num_active_point,
                                        Int num_block,
                                        Int max_point_per_block,
                                        Int max_input_address,
                                        Int ctr,
                                        Int use_normal)  {
    // iterate over all ouput points
    for(Int tid = blockIdx.x * blockDim.x + threadIdx.x;
                            tid < num_active_point;
                            tid += blockDim.x * gridDim.x)  
    {
        Int blockid = d_active_output_point_blockid[tid];
        
        Int offset_start = d_block_input_address_list_size[blockid];
        Int offset_end = d_block_input_address_list_size[blockid+1];
        
        if(offset_end - offset_start > max_input_address)   {
            continue;
        }

        Int rulebook_id = atomicAdd(&d_block_active_output_count[blockid], 1);
        Int rulebook_count = 0;
        Int pre_pos = 0;
        for(Int i = 0; i < 27; i++)  {
            if(d_input_point_query_result[tid * 27 + i] == -1)  // Not active
                continue;

            /***  get local id ***/
            Int idx = -1;
            // time consuming?
            // find point corresbonding compact index.
            for(Int j = pre_pos; j + offset_start < offset_end; j++) {
                if(d_block_input_address_list[offset_start + j] == d_input_point_query_result[tid * 27 + i])    {
                    idx = j;
                    pre_pos = j + 1;
                    break;
                }
            }
            /*** End ***/
            
            assert(idx != -1);

            /* Rulebook (require further compacting)*/ 
            Int conv_pos = i;
            if(use_normal)
                conv_pos = rot_index[27 * d_ori_index[tid] + i];

            assert(rulebook_id <= max_input_address);
            // index : block, point, input region   val : input offset in compact input addresses
            d_output_rulebook[(blockid * max_input_address * 27) + (rulebook_id * 27) + conv_pos] = (short)idx; // input address offset
            rulebook_count++;
            
        }
        // /* conv_structure */
        // output address of [block, in_block_loc]
        d_output_output_address[blockid * max_point_per_block + rulebook_id] = d_input_point_query_result[tid * 27 + 13] + ctr;
    }
}

__global__
void _dGenerateInputAddress(Int* d_block_input_address_list,
                            Int* d_output_input_address,
                            Int* d_block_input_address_list_size,
                            Int num_block,
                            Int ctr)
{
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < d_block_input_address_list_size[num_block])  {
        d_output_input_address[tid] = d_block_input_address_list[tid] + ctr;
        tid += blockDim.x * gridDim.x;
    }
}

void dGenerateChuckRulebook (Int* d_input_point_query_result,             // size = num_active_point * 27
                            Int* d_ori_index,                            // size = num_active_point
                            Int* d_active_output_point_blockid,          // size = num_active_point
                            Int* d_block_active_output_count,            /* initialize with 0 */

                            Int* d_block_input_address_list,
                            Int* d_block_input_address_list_size,

                            short* d_output_rulebook,                      // dim = (2, num_block, max_input_address * 27) 
                            
                            Int* d_output_output_address,                // size = [num_block, max_input_address]

                            Int* d_output_input_address,

                            Int num_active_point,
                            Int num_block,
                            Int max_point_per_block,
                            Int max_input_address,
                            Int ctr,
                            Int use_normal)
{

    /* Launch kernel */
    _dGenerateChuckRulebook<<<512,512>>> (d_input_point_query_result,
                                        d_ori_index,
                                        d_active_output_point_blockid,
                                        d_block_active_output_count, 
                                        d_block_input_address_list,
                                        d_block_input_address_list_size,           
                                        d_output_rulebook,                      
                                        d_output_output_address,                
                                        num_active_point,
                                        num_block,
                                        max_point_per_block,
                                        max_input_address,
                                        ctr,
                                        use_normal);

#ifdef KERNEL_DEBUG
    printf("dGenerateChuckRulebook\n");
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#endif
    // generate output input address which considers batch offset.
  _dGenerateInputAddress<<<512,512>>> (d_block_input_address_list,
                                      d_output_input_address,
                                      d_block_input_address_list_size,
                                      num_block,
                                      ctr);

#ifdef KERNEL_DEBUG
    printf("dGenerateChuckRulebook\n");
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#endif
}



__global__ void _dGenerateIncreChuckRulebook(Int* d_input_point_query_result,             // size = num_active_point * 27
                                        Int *d_ori_index,                               // size = num_active_point
                                        Int* d_output_point_query_result,                      // size = num_active_point
                                        Int* d_active_output_point_blockid,          // size = num_active_point
                                        Int* d_block_active_output_count,            /* initialize with 0 */

                                        Int* d_block_input_address_list,
                                        Int* d_block_input_address_list_size,
                                        Int* d_chunk_points_num,
                                        short* d_output_rulebook,                      // dim = (2, num_block, max_input_address * 27) 
                                        Int* d_output_output_address,                // size = [num_block, max_input_address]

                                        Int num_active_point,
                                        Int num_block,
                                        Int max_point_per_block,
                                        Int max_input_address,
                                        Int ctr,
                                        Int use_normal)  {
    // iterate over all ouput points
    for(Int tid = blockIdx.x * blockDim.x + threadIdx.x;
                            tid < num_active_point;
                            tid += blockDim.x * gridDim.x)  
    {
        Int blockid = d_active_output_point_blockid[tid];
        
        Int offset_start = d_block_input_address_list_size[blockid];
        Int offset_end = d_block_input_address_list_size[blockid+1];
        
        if(offset_end - offset_start > max_input_address or d_chunk_points_num[blockid] > max_input_address)   {
            continue;
        }

        Int rulebook_id = atomicAdd(&d_block_active_output_count[blockid], 1);
        Int rulebook_count = 0;
        Int pre_pos = 0;
        for(Int i = 0; i < 27; i++)  {
            if(d_input_point_query_result[tid * 27 + i] == -1)  // Not active
                continue;

            /***  get local id ***/
            Int idx = -1;
            // time consuming?
            // find point corresbonding compact index.
            for(Int j = 0; j + offset_start < offset_end; j++) {
                if(d_block_input_address_list[offset_start + j] == d_input_point_query_result[tid * 27 + i])    {
                    idx = j;
                    //pre_pos = j + 1;
                    break;
                }
            }
            /*** End ***/
            
            assert(idx != -1);

            /* Rulebook (require further compacting)*/ 
            Int conv_pos = i;
            if(use_normal)
                conv_pos = rot_index[27 * d_ori_index[tid] + i];
            
            assert(rulebook_id <= max_input_address);
            // index : block, point, input region   val : input offset in compact input addresses
            d_output_rulebook[(blockid * max_input_address * 27) + (rulebook_id * 27) + conv_pos] = (short)idx; // input address offset
            rulebook_count++;
            
        }
        // /* conv_structure */
        // output address of [block, in_block_loc]
       // assert(d_output_point_query_result[tid] >= 0);
        d_output_output_address[blockid * max_point_per_block + rulebook_id] = d_output_point_query_result[tid];
    }
}

void dGenerateIncreChuckRulebook (Int* d_input_point_query_result,             // size = num_active_point * 27
                            Int* d_ori_index,                            // size = num_active_point
                            Int* d_output_point_query_result,                      // size = num_active_point
                            Int* d_active_output_point_blockid,          // size = num_active_point
                            Int* d_block_active_output_count,            /* initialize with 0 */

                            Int* d_block_input_address_list,
                            Int* d_block_input_address_list_size,
                            Int* d_chunk_points_num,
                            short* d_output_rulebook,                      // dim = (2, num_block, max_input_address * 27) 
                            
                            Int* d_output_output_address,                // size = [num_block, max_input_address]

                            Int* d_output_input_address,

                            Int num_active_point,
                            Int num_block,
                            Int max_point_per_block,
                            Int max_input_address,
                            Int ctr,
                            Int use_normal)
{

    /* Launch kernel */
    _dGenerateIncreChuckRulebook<<<512,512>>> (d_input_point_query_result,
                                        d_ori_index,
                                        d_output_point_query_result,
                                        d_active_output_point_blockid,
                                        d_block_active_output_count, 
                                        d_block_input_address_list,
                                        d_block_input_address_list_size,   
                                        d_chunk_points_num,        
                                        d_output_rulebook,                      
                                        d_output_output_address,                
                                        num_active_point,
                                        num_block,
                                        max_point_per_block,
                                        max_input_address,
                                        0,
                                        use_normal);

    _dGenerateInputAddress<<<512,512>>> (d_block_input_address_list,
                                        d_output_input_address,
                                        d_block_input_address_list_size,
                                        num_block,
                                        0);

}



__global__ void _dgetSplitPointListFlag(uint32_t* d_out_flag,            // size = NActivePoints
                                        Int* d_blockid,              // size = NActivePoints
                                        Int* d_output_input_address_size,              // size = num_block+1
                                        Int NActivePoints,
                                        Int Threshold)   
{
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < NActivePoints)  {
        d_out_flag[tid] = ((d_output_input_address_size[d_blockid[tid]+1] - d_output_input_address_size[d_blockid[tid]]) > Threshold) ? 1 : 0;
        tid += blockDim.x * gridDim.x;
    }
}

void dgetSplitPointListFlag(uint32_t* d_out_flag,    
                            Int* d_blockid,     
                            Int* d_block_input_address_list_size,    
                            Int NActivePoints,
                            Int Threshold)   
{

    _dgetSplitPointListFlag<<<512,512>>>(d_out_flag,
                                        d_blockid,     
                                        d_block_input_address_list_size,
                                        NActivePoints,
                                        Threshold);

#ifdef KERNEL_DEBUG
    printf("dGenerateChuckRulebook\n");
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#endif
}

__global__ void _dIncreGetSplitPointListFlag(uint32_t* d_out_flag,            // size = NActivePoints
                                        Int* d_blockid,              // size = NActivePoints
                                        Int* d_output_input_address_size,              // size = num_block+1
                                        Int* d_chunk_points_num,   
                                        Int NActivePoints,
                                        Int Threshold)   
{
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < NActivePoints)  {
        d_out_flag[tid] = ((d_output_input_address_size[d_blockid[tid]+1] - d_output_input_address_size[d_blockid[tid]]) > Threshold or d_chunk_points_num[d_blockid[tid]] > Threshold) ? 1 : 0;
        tid += blockDim.x * gridDim.x;
    }
}

void dIncreGetSplitPointListFlag(uint32_t* d_out_flag,    
                            Int* d_blockid,     
                            Int* d_block_input_address_list_size, 
                            Int* d_chunk_points_num,   
                            Int NActivePoints,
                            Int Threshold)   
{

    _dIncreGetSplitPointListFlag<<<512,512>>>(d_out_flag,
                                        d_blockid,     
                                        d_block_input_address_list_size,    
                                        d_chunk_points_num,   
                                        NActivePoints,
                                        Threshold);

#ifdef KERNEL_DEBUG
    printf("dGenerateChuckRulebook\n");
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#endif
}


__global__ 
void _dGetChunkQueryList(Int* d_block_grids,
                        Int num_block,
                        Int chunk_size,
                        Int* d_chunk_query_list,
                        Int d_chunk_query_list_size)
{
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Int max_point_per_block = (chunk_size+2)*(chunk_size+2)*(chunk_size+2);
    uint32_t hash;
    while(tid < num_block)  {
        Int x = chunk_size * d_block_grids[tid];
        Int y = chunk_size * d_block_grids[tid + num_block];
        Int z = chunk_size * d_block_grids[tid + 2*num_block];
        Int idx = 0;

        //maybe need x > 1, y > 1, z > 1?
#pragma unroll
        for(Int i = x-1; i < x+chunk_size+1; i++)  {
#pragma unroll
            for(Int j = y-1; j < y+chunk_size+1; j++)  {
#pragma unroll
                for(Int k = z-1; k < z+chunk_size+1; k++)  {
                    hash = (k << 21 | j << 10 | i);
                    hash &= 0x7FFFFFFF;
                    d_chunk_query_list[tid * max_point_per_block + idx] = hash;
                    idx++;
                }
            }
        }
        tid += blockDim.x * gridDim.x;
    }
}

//get chunk input location(in original coords), save to d_chunk_query_list(chunk num * (chunksize+2)^3)
void dGetChunkQueryList(Int* d_block_grids,
                        Int num_block,
                        Int chunk_size,
                        Int* d_chunk_query_list,
                        Int d_chunk_query_list_size)
{
    _dGetChunkQueryList<<<512,512>>>(d_block_grids,
                                    num_block,
                                    chunk_size,
                                    d_chunk_query_list,
                                    d_chunk_query_list_size);
}

__global__ 
void _dGetFlag(Int* d_chunk_result_list, 
                uint32_t* d_chunk_result_flag, 
                Int* d_chunk_result_count,
                Int num_block,
                Int max_point_per_block)    {
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < num_block * max_point_per_block)  {
        if (d_chunk_result_list[tid] == -1) {
            d_chunk_result_flag[tid] = 0;
        }
        else{
            atomicAdd(&d_chunk_result_count[tid / max_point_per_block], 1);
            d_chunk_result_flag[tid] = 1;
        }
        tid += blockDim.x * gridDim.x;
    }
}

void dGetFlag(Int* d_chunk_result_list, 
            uint32_t* d_chunk_result_flag,
            Int* d_chunk_result_count,
            Int num_block,
            Int max_point_per_block)
{
    _dGetFlag<<<512,512>>>(d_chunk_result_list, 
                            d_chunk_result_flag, 
                            d_chunk_result_count,
                            num_block,
                            max_point_per_block);
}
__global__
void _dGenerateChunkBackwardRB(
    Int* d_input_point_query_result,             // size = num_active_point * 27
    Int* d_ori_index,                            // size = num_active_point
    Int* d_active_output_point_blockid,          // size = num_active_point
    Int* d_block_output_address,                 // size = num_block * max_input_address
    Int* d_block_active_output_count,            // size = num_block
    Int* d_block_input_address_list,             // size = num_block
    Int* d_block_input_address_list_size,        // size = num_block + 1

    short* d_output_rulebook_backward,             // size = num_block * max_input_address * 27, initialized with -1

    Int num_active_point,
    Int num_block,
    Int max_point_per_block,
    Int max_input_address,
    Int ctr,
    Int use_normal
)
{
    /* for each active input I, get its corresponding output O
     * 1. get its block id, conv_position
     * 2. in the scope of that block, search for local id for I & O
     * 3. add rule to rulebook, at (blockid, I's local_id, conv_position)
     */

    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < num_active_point * 27)  {
        Int global_input_address = d_input_point_query_result[tid];
        if (global_input_address == -1) {
            tid += blockDim.x * gridDim.x;
            continue;
        }
        Int global_output_id = tid / 27;
        Int global_output_address = d_input_point_query_result[global_output_id * 27 + 13];
        // Int global_input_address = tid / 27;
        Int conv_pos = tid % 27;

        if(use_normal)
        conv_pos = rot_index[27 * d_ori_index[global_output_address] + conv_pos];

        Int block_id = d_active_output_point_blockid[global_output_id];

        // looking for input
        Int offset_start = d_block_input_address_list_size[block_id];
        Int offset_end = d_block_input_address_list_size[block_id+1];

        if (offset_end - offset_start > max_input_address) {
            tid += blockDim.x * gridDim.x;
            continue;
        }

        Int idx_input = -1;
        for(Int j = 0; j + offset_start < offset_end; j++) {
            if(d_block_input_address_list[offset_start + j] == global_input_address)    {
                idx_input = j;
            }
        }
        
        // looking for output
        Int d_block_output_address_size = d_block_active_output_count[block_id];
        Int idx_output = -1;
        for (Int i = 0; i < d_block_output_address_size; i++)   {
            if(d_block_output_address[block_id * max_point_per_block + i] - ctr == global_output_address)    {
                idx_output = i;
            }
        }

        assert(idx_input != -1);
        assert(idx_output != -1);

        d_output_rulebook_backward[block_id * max_input_address * 27 + idx_input * 27 + conv_pos] = (short)idx_output;

        tid += blockDim.x * gridDim.x;
    }
}


void dGenerateChunkBackwardRB(
    Int* d_input_point_query_result,             // size = num_active_point * 27
    Int* d_ori_index,                            // size = num_active_point
    Int* d_active_output_point_blockid,          // size = num_active_point
    Int* d_block_output_address,                 // size = num_block * max_input_address
    Int* d_block_active_output_count,            // size = num_block
    Int* d_block_input_address_list,             // size = num_block
    Int* d_block_input_address_list_size,        // size = num_block + 1
    
    short* d_output_rulebook_backward,                      // size = num_block * max_input_address * 27
    
    Int num_active_point,
    Int num_block,
    Int max_point_per_block,
    Int max_input_address,
    Int ctr,
    Int use_normal
)
{
    _dGenerateChunkBackwardRB<<<512,512>>>(
        d_input_point_query_result,             // size = num_active_point * 27
        d_ori_index,                            // size = num_active_point
        d_active_output_point_blockid,          // size = num_active_point
        d_block_output_address,                 // size = num_block * max_input_address
        d_block_active_output_count,            // size = num_block
        d_block_input_address_list,             // size = num_block
        d_block_input_address_list_size,        // size = num_block + 1
        
        d_output_rulebook_backward,                      // size = num_block * max_input_address * 27
        
        num_active_point,
        num_block,
        max_point_per_block,
        max_input_address,
        ctr,
        use_normal
    );
}

#if 1

__global__ void _d_Convolution_GenerateOutputRules(uint32_t * d_in_points, uint32_t * d_output_points,
                                                   uint32_t * d_output_index, uint32_t ** d_address_list,
                                                   Int input_points_num, Int dimension,Int filterSize,
                                                   Int input_offset, Int convElementNum)
{
    extern __shared__ uint32_t * address_list[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x < convElementNum * 2)
    {
        address_list[threadIdx.x] = d_address_list[threadIdx.x];
    }

    __syncthreads();
    // kernel size 2*2*2, stride 2 is a special case that input points num equals to rule num. so that below is correct.
    while(tid < input_points_num)
    {
        int data = 0;
        //  data is input region offset.
        for(int i = 0; i < dimension; i++)
        {
            data = data * filterSize;
            data += (d_in_points[tid + input_points_num * i] - d_output_points[tid + input_points_num * i] * 2);
        }
        // if(data > convElementNum -1 || data < 0)
        // {
        //     for(int i = 0; i < dimension; i++)
        //     {
        //         printf("%d %d %d %d\r\n",dimension,data,d_in_points[tid + input_points_num * i], d_output_points[tid + input_points_num * i]);
        //     }
        // }
        address_list[data][tid * 2] = tid + input_offset;
        address_list[data][tid * 2 + 1] = d_output_index[tid];
        address_list[data + convElementNum][tid] = 1;
//        address_list[data + convElementNum][tid * 2 + 1] = 1;
        tid += blockDim.x * gridDim.x;
    }
}

void d_Convolution_GenerateOutputRules(uint32_t * d_in_points, uint32_t * d_output_points, uint32_t * d_output_index,
                                       RuleBook &rules,Int num,  Int dimension, Int filterSize, Int input_offset)
{
    int convElementNum = rules.size();
    uint32_t ** address_list = new uint32_t *[convElementNum * 2];
    uint32_t ** d_address_list = NULL;
    gpuErrchk(cudaMalloc((void**)&d_address_list, sizeof(uint32_t * ) * convElementNum * 2));

    for(int i = 0; i < convElementNum; i++)
    {
        uint32_t *d_rules_temp = NULL;
        uint32_t *d_rules_flag_temp = NULL;
        gpuErrchk(cudaMalloc((void**)&d_rules_temp, sizeof(uint32_t) * num * 2));
        gpuErrchk(cudaMalloc((void**)&d_rules_flag_temp, sizeof(uint32_t) * num));
        gpuErrchk(cudaMemset(d_rules_flag_temp, 0, sizeof(uint32_t) * (num)));
        address_list[i] = d_rules_temp;
        address_list[i + convElementNum] = d_rules_flag_temp;
    }
    gpuErrchk(cudaMemcpy(d_address_list, address_list, sizeof(uint32_t *) * convElementNum * 2, cudaMemcpyHostToDevice));


    _d_Convolution_GenerateOutputRules<<<512,512,convElementNum * sizeof(uint32_t * )*2>>>
                (d_in_points, d_output_points, d_output_index, d_address_list,
                 num,dimension,filterSize,input_offset,convElementNum);


    size_t rules_num;

    uint32_t * d_generated_rules = NULL;
    gpuErrchk(cudaMalloc((void**)&d_generated_rules, sizeof(uint32_t) * num * 2));
    size_t* h_num = new size_t[1];
    size_t* d_num;
    gpuErrchk(cudaMalloc((void **)&d_num, sizeof(size_t)  * convElementNum));
    CUDPP_Compacting convCompact(num ,CUDPP_ULONGLONG);

    for(int i = 0; i < convElementNum; i++)
    {
        convCompact.apply(d_generated_rules, &d_num[i], address_list[i], address_list[i+ convElementNum], num);
        gpuErrchk(cudaMemcpy(h_num, &d_num[i], sizeof(size_t), cudaMemcpyDeviceToHost));

        Ints & rule = rules[i];
        rules_num = *h_num;
        rule.resize(rule.size() + rules_num * 2);
        gpuErrchk(cudaMemcpy(rule.data() + rule.size() - rules_num * 2, d_generated_rules, sizeof(Int) * 2 * rules_num, cudaMemcpyDeviceToHost));

    }

    gpuErrchk(cudaFree(d_generated_rules));
    gpuErrchk(cudaFree(d_address_list));
    gpuErrchk(cudaFree(d_num));

    for(int i = 0; i < convElementNum; i++)
    {
        gpuErrchk(cudaFree(address_list[i]));
        gpuErrchk(cudaFree(address_list[i + convElementNum]));
    }
    delete address_list;
    delete h_num;
}
#endif
#ifdef NEW_SPTIAL_POINT 

__global__ void _dGenerateSpatialNewPoint (Int* d_prev_all_point,             // size = num_active_point * dim
	Int* d_next_all_point,                            // size = num_active_point * maxi_sizec * dim
	long* size, //converted to cuda mem
    long* stride,
    long* output_spatial_size,
    Int num_active_point,
    long maxi_sizec,
	Int ndim)
{
Int tid = blockIdx.x * blockDim.x + threadIdx.x;
while(tid < num_active_point)
{
    Int* p= new Int[ndim];
    Int* lb= new Int[ndim];
    Int* ub= new Int[ndim];
    for(Int k = 0; k < ndim; k++)  {
        p[k] = d_prev_all_point[tid + k * num_active_point];
        lb[k] = (Int)max((unsigned int)0, (unsigned int)((p[k] - size[k] + stride[k]) / stride[k]));
        ub[k] = (Int)min((unsigned int)output_spatial_size[k] - 1, (unsigned int)(p[k] / stride[k]));        
    }
    // if(tid==0)
    // {
    //     // printf("p: %d,%d,%d\n",p[0],p[1],p[2]);
    //     // printf("lb: %d,%d,%d\n",lb[0],lb[1],lb[2]);
    //     // printf("ub: %d,%d,%d\n",ub[0],ub[1],ub[2]);

    // }
    // For efficiency, support limited ndim here.
    // To support infini ndim, recusive func should be used here
    assert(ndim==3);
    if (ndim==3)
    {
        Int i,j,k,out_count=0;
        // for(i=0;i<maxi_sizec;i++)
        // {

        // }
        for(i=lb[0];i<=ub[0];i++)
        for(j=lb[1];j<=ub[1];j++)
        for(k=lb[2];k<=ub[2];k++)
        {
            d_next_all_point[(tid + 0 * num_active_point)*maxi_sizec+out_count]=i;
            d_next_all_point[(tid + 1 * num_active_point)*maxi_sizec+out_count]=j;
            d_next_all_point[(tid + 2 * num_active_point)*maxi_sizec+out_count]=k;
            out_count++;
        }

    }
    
    tid+= blockDim.x * gridDim.x;
    delete p;
    delete lb;
    delete ub;
}
}

void dGenerateSpatialNewPoint (Int* d_prev_all_point,             // size = num_active_point * dim
	Int* d_next_all_point,                            // size = num_active_point * maxi_sizec * dim
	long* d_size,
    long* d_stride,
    long* d_output_spatial_size,
    Int num_active_point,
    long maxi_sizec,
	Int ndim)
    {
        // printf("skip");
        _dGenerateSpatialNewPoint<<<512,512>>> (
        d_prev_all_point,             // size = num_active_point * dim
        d_next_all_point,                            // size = num_active_point * maxi_sizec * dim
        d_size,
        d_stride,
        d_output_spatial_size,
        num_active_point,
        maxi_sizec,
        ndim);
    }

__global__ void _dGetNot(Int *a, Int num) {
    Int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while(tid < num) {
        a[tid] = a[tid] >= 0 ? 0 : 1;
        tid += blockDim.x * gridDim.x;
    }
}


void dGetNot(Int *a, Int num) {
    _dGetNot<<<512, 512>>>(a, num);
}

__global__ void _dGetChunkPointsNum(Int *chunk_ids, Int *chunk_num, Int points_num) {
    Int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while(tid < points_num) {
        atomicAdd(&chunk_num[chunk_ids[tid]], 1);
        tid += blockDim.x * gridDim.x;
    }
}

void dGetChunkPointsNum(Int *chunk_ids, Int *chunk_num, Int points_num) {
    _dGetChunkPointsNum<<<512, 512>>>(chunk_ids, chunk_num, points_num);
}


#endif


