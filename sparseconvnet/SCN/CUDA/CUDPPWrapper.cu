// Added by Tian Zheng
// t-zheng@outlook.com
// cuda kernels and launchers

#define HASH_NOT_FOUND 0xFFFFFFFF





__global__ void _dFlatPoints(uint32_t* d_input_points, uint32_t * d_output_points, int dimension, int size)    {
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < size)  {
        for(int i = 0; i < dimension; i++)
        {
            d_output_points[tid + size * i] = d_input_points[tid * dimension + i];
        }
        tid += blockDim.x * gridDim.x;
    }
}

at::Tensor FlatPoints(const at::Tensor &input_points)
{
    at::Tensor flatten_points = at::empty({input_points.size(0),input_points.size(1)}, at::CUDA(at_kINT));
    int dimension = input_points.size(1);
    int points_num = input_points.size(0);

    _dFlatPoints<<<512,512>>>((uint32_t*)input_points.data<Int>(), (uint32_t*)flatten_points.data<Int>(),
                              dimension, points_num);
    return flatten_points;
}

// eight times are required for each pixel? Too complicated, they are querying the same value!



/* Basic Hash*/
__global__ void _dGetRepeatNums(uint2 *d_index_counts, uint32_t* d_repeatNums, int size)    {
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < size)  {
        d_repeatNums[tid] = d_index_counts[tid].y;
        tid += blockDim.x * gridDim.x;
    }
}


void GetRepeatNums(uint2 *d_index_counts, uint32_t* d_repeatNums, int size)
{
    _dGetRepeatNums<<<512,512>>>(d_index_counts, d_repeatNums,size);
}


__global__ void _dCollectInputRules(uint2 * d_index_counts, uint32_t * d_all_values, Int * d_input_rules,
                                    int size, int max_repeat_num)    {
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < size)  {
        d_input_rules[tid * (max_repeat_num + 1)] = d_index_counts[tid].y;
        for(int i = 0; i < d_index_counts[tid].y; i++)
        {
            d_input_rules[tid * (max_repeat_num + 1) + i + 1] = d_all_values[d_index_counts[tid].x + i];
        }
        tid += blockDim.x * gridDim.x;
    }
}


void CollectInputRules(uint2 * d_index_counts,uint32_t * d_all_values, Int * d_input_rules, int size, int max_repeat_num)
{
    _dCollectInputRules<<<512,512>>>(d_index_counts, d_all_values, d_input_rules, size, max_repeat_num);
}

__global__ void _dPreprocessPointClouds(long *d_ori_points ,uint32_t* d_points,  uint32_t* d_keys, uint32_t* d_index, uint32_t *d_point_cloud_start ,
                                        const Int batch_size, const int nInputRows){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < nInputRows - 1){

        d_points[nInputRows * 0 + tid] = d_ori_points[tid*4];
        d_points[nInputRows * 1 + tid] = d_ori_points[tid*4 + 1];
        d_points[nInputRows * 2 + tid] = d_ori_points[tid*4 + 2];
        uint32_t hash = (d_ori_points[tid*4 + 2] << 21 | d_ori_points[tid*4 + 1] << 10 | d_ori_points[tid*4]);
        hash &= 0x7FFFFFFF;
        d_index[tid] = tid;
        d_keys[tid] = hash;
        if(d_ori_points[tid * 4 + 3] < d_ori_points[tid*4 + 7])
        {
            d_point_cloud_start[d_ori_points[tid * 4 + 3] + 1] = tid + 1;
        }
        tid += blockDim.x * gridDim.x;
    }
    if(tid == nInputRows - 1)
    {
        d_points[nInputRows * 0 + tid] = d_ori_points[tid*4];
        d_points[nInputRows * 1 + tid] = d_ori_points[tid*4 + 1];
        d_points[nInputRows * 2 + tid] = d_ori_points[tid*4 + 2];
        uint32_t hash = (d_ori_points[tid*4 + 2] << 21 | d_ori_points[tid*4 + 1] << 10 | d_ori_points[tid*4]);
        hash &= 0x7FFFFFFF;
        d_index[tid] = tid;
        d_keys[tid] = hash;
    }
}


void PreprocessPointClouds(long *d_ori_points ,uint32_t* d_points,  uint32_t* d_keys, uint32_t* d_index, uint32_t *d_point_cloud_start , const Int batch_size,
                               const int nInputRows)
{
        _dPreprocessPointClouds<<<512,512>>> (d_ori_points , d_points, d_keys, d_index, d_point_cloud_start , batch_size, nInputRows);
}

/* Basic Hash*/
__global__ void _dGetPointHash(uint32_t* d_points, uint32_t* d_keys, uint32_t* d_index, Int num)    {
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < num)  {
        uint32_t hash = (d_points[tid + num * 2] << 21 | d_points[tid + num] << 10 | d_points[tid]);
        hash &= 0x7FFFFFFF;
        d_index[tid] = tid;
        d_keys[tid] = hash;
        tid += blockDim.x * gridDim.x;
    }
}

// __global__ void _dGetPointHashQuery(uint32_t* d_points, uint32_t* d_keys, Int num)    {
//     Int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     while(tid < num)  {
//         uint32_t hash = 16777619;
//         for (Int i = 0; i < 3; i++) {
//             hash *= 2166136261;
//             hash ^= d_points[num * i + tid];
//         }
//         d_keys[tid] = hash;
//         tid += blockDim.x * gridDim.x;
//     }
// }

__global__ void _dGetPointHashQuery(uint32_t* d_points, uint32_t* d_keys, Int num)    {
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < num)  {
        uint32_t hash = (d_points[tid + num * 2] << 21 | d_points[tid + num] << 10 | d_points[tid]);
        hash &= 0x7FFFFFFF;
        d_keys[tid] = hash;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void _dMultivalHashGetValue (Int* _d_hash_idx,           // size = hashtable_size
                                        uint32_t* d_all_values,     // size = hashtable_size
                                        uint2* d_query_vals_multivalue,  // size = query_size
                                        uint32_t* d_results,             // size = query_size
                                        Int query_size,
                                        Int hashtable_size)
{
    // input key  output val
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < query_size)  {
        uint2 current_query_vals_multivalue = d_query_vals_multivalue[tid];
        d_results[tid] = current_query_vals_multivalue.y > 0 ? 
                    _d_hash_idx[d_all_values[current_query_vals_multivalue.x]]
                    : HASH_NOT_FOUND;
        tid += blockDim.x * gridDim.x;
    }
}


void dGetPointHash(uint32_t* d_points, uint32_t* d_keys, uint32_t* d_index, Int num)   {
    _dGetPointHash<<<512,512>>> (d_points, d_keys, d_index, num);
}

void dGetPointHashQuery(uint32_t* d_points, uint32_t* d_keys, Int num)   {
    _dGetPointHashQuery<<<512,512>>> (d_points, d_keys, num);
}

void dMultivalHashGetValue(Int* _d_hash_idx, 
                            uint32_t* d_all_values, 
                            uint2* d_query_vals_multivalue,
                            uint32_t* d_results,
                            Int query_size,
                            Int hashtable_size)
{
    _dMultivalHashGetValue<<<512,512>>>(_d_hash_idx,
                                        d_all_values, 
                                        d_query_vals_multivalue,
                                        d_results,
                                        query_size,
                                        hashtable_size);
}

/* Compacting */
__global__ void _dCopyUniquePoints(uint32_t* _d_hash_points, // size = 3 * size
                                   uint32_t* d_points,       // size = 3 * num
                                   uint32_t* d_index, 
                                //    uint32_t* d_index_count,
                                   Int num,
                                   Int size)    {
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < num)  {
        Int idx = d_index[tid];
        // atomicAdd(&d_index_count[idx], 1);
        // is there a write conflict?
        _d_hash_points[idx] = d_points[tid];
        _d_hash_points[idx + size] = d_points[tid + num];
        _d_hash_points[idx + size * 2] = d_points[tid + num * 2];
        tid += blockDim.x * gridDim.x;
    }
}


void dCopyUniquePoints(uint32_t* _d_hash_points, uint32_t* d_points, uint32_t* d_index, Int num, Int size)    {
    _dCopyUniquePoints<<<512,512>>>(_d_hash_points, d_points, d_index, num, size);
}



__global__ void _dMultivalPointHashtableInsertHelper(
    uint2* d_index_counts,      // size = index_counts_size
	uint32_t* d_all_values,     // size = all_values_size
    uint32_t* d_points,         // size = 3 * all_values_size
    Int* _d_hash_idx,           // size = all_values_size
    Int* _d_hash_points,        // size = 3 * index_counts_size
    uint32_t index_counts_size,
    uint32_t all_values_size)
{
    Int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < index_counts_size)  {
        for(Int i = 0; i < d_index_counts[tid].y; i++)  {
            _d_hash_idx[d_all_values[d_index_counts[tid].x + i]] = tid;
        }
        _d_hash_points[tid] = d_points[d_all_values[d_index_counts[tid].x]];
        _d_hash_points[tid + index_counts_size] = d_points[d_all_values[d_index_counts[tid].x] + all_values_size];
        _d_hash_points[tid + 2*index_counts_size] = d_points[d_all_values[d_index_counts[tid].x] + 2*all_values_size];
        tid += blockDim.x * gridDim.x;
    }
}



void dMultivalPointHashtableInsertHelper(
	uint2* d_index_counts,
	uint32_t* d_all_values,
	uint32_t* d_points,
	Int* _d_hash_idx,
	Int* _d_hash_points,
	uint32_t index_counts_size,
	uint32_t all_values_size)
{
    _dMultivalPointHashtableInsertHelper<<<512,512>>>(
        d_index_counts,      // size = index_counts_size
        d_all_values,     // size = all_values_size
        d_points,         // size = 3 * all_values_size
        _d_hash_idx,           // size = all_values_size
        _d_hash_points,        // size = 3 * index_counts_size
        index_counts_size,
        all_values_size);
}
