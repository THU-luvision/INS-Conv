#ifndef CUDPPWRAPPER_HPP
#define CUDPPWRAPPER_HPP

#include <cstdio>
#include <cudpp.h>
#include <cudpp_hash.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include "../Metadata/Metadata.h"

#define HASH_NOT_FOUND 0xFFFFFFFF

/********************/
/* CUDA ERROR CHECK */
/********************/
#ifndef gpuErrchk
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

/* CUDPP */ 
class CUDPP_Compacting  {
public:
    CUDPPHandle theCudpp;
    CUDPPConfiguration config;
    CUDPPHandle plan;

    CUDPP_Compacting(size_t size, CUDPPDatatype dtype = CUDPP_INT)  {
        config.algorithm = CUDPP_COMPACT;
        config.datatype = dtype;
        config.options = CUDPP_OPTION_FORWARD;
        CUDPPResult result;
        result = cudppCreate(&theCudpp);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Compacting cudppCreate()\n");
        }
        result = cudppPlan(theCudpp, &plan, config, size, 1, 0);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Compacting cudppPlan()\n");
        }
    }

    void apply(void* d_out, size_t* d_numValidElements, const void *d_in, const unsigned int *d_isValid, size_t numElements)    {
        EASY_FUNCTION(profiler::colors::Brown200);
        
        CUDPPResult result = cudppCompact(plan, d_out, d_numValidElements, d_in, d_isValid, numElements);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppCompact()\n");
            abort();
        }

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif

    }

    ~CUDPP_Compacting() {
        CUDPPResult result;
        result = cudppDestroyPlan(plan);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppDestroyPlan()\n");
        }
        result = cudppDestroy(theCudpp);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppDestroy()\n");
        }
    }
};

class CUDPP_Scan  {
public:
    CUDPPHandle theCudpp;
    CUDPPConfiguration config;
    CUDPPHandle plan;

    CUDPP_Scan(size_t size)  {
        config.algorithm = CUDPP_SCAN;
        config.datatype = CUDPP_INT;
        config.op = CUDPP_ADD;

        CUDPPOption direction = CUDPP_OPTION_FORWARD;
        CUDPPOption inclusivity = CUDPP_OPTION_EXCLUSIVE;

        config.options = direction | inclusivity;

        CUDPPResult result;
        result = cudppCreate(&theCudpp);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Scan cudppCreate()\n");
        }
        result = cudppPlan(theCudpp, &plan, config, size, 1, 0);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Scan cudppPlan()\n");
        }
    }

    void apply(void* d_out, const void *d_in, size_t numElements)    {
        CUDPPResult result = cudppScan(plan, d_out, d_in, numElements);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppScan()\n");
            abort();
        }
    }

    ~CUDPP_Scan() {
        CUDPPResult result;
        result = cudppDestroyPlan(plan);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppDestroyPlan()\n");
        }
        result = cudppDestroy(theCudpp);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppDestroy()\n");
        }
    }
};

class CUDPP_Segmented_Scan  {
public:
    CUDPPHandle theCudpp;
    CUDPPConfiguration config;
    CUDPPHandle plan;

    CUDPP_Segmented_Scan(size_t size)  {
        config.algorithm = CUDPP_SEGMENTED_SCAN;
        config.datatype = CUDPP_INT;
        config.op = CUDPP_ADD;

        CUDPPOption direction = CUDPP_OPTION_FORWARD;
        CUDPPOption inclusivity = CUDPP_OPTION_EXCLUSIVE;

        config.options = direction | inclusivity;

        CUDPPResult result;
        result = cudppCreate(&theCudpp);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Scan cudppCreate()\n");
        }
        result = cudppPlan(theCudpp, &plan, config, size, 1, 0);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Scan cudppPlan()\n");
        }
    }

    void apply(void* d_out, const void *d_in, const unsigned int * d_iflags, size_t numElements)    {
        CUDPPResult result = cudppSegmentedScan(plan, d_out, d_in, d_iflags, numElements);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppSegmentedScan()\n");
            abort();
        }
    }

    ~CUDPP_Segmented_Scan() {
        CUDPPResult result;
        result = cudppDestroyPlan(plan);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppDestroyPlan()\n");
        }
        result = cudppDestroy(theCudpp);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppDestroy()\n");
        }
    }
};

class CUDPP_Reduce  {
public:
    CUDPPHandle theCudpp;
    CUDPPConfiguration config;
    CUDPPHandle plan;

    CUDPP_Reduce(size_t size)  {
        config.algorithm = CUDPP_REDUCE;
        config.datatype = CUDPP_INT;
        config.op = CUDPP_MAX;
        config.options = 0;

        CUDPPResult result;
        result = cudppCreate(&theCudpp);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Scan cudppCreate()\n");
        }
        result = cudppPlan(theCudpp, &plan, config, size, 1, 0);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Scan cudppPlan()\n");
        }
    }

    void apply(void* d_out, const void *d_in, size_t numElements)    {
        CUDPPResult result = cudppReduce(plan, d_out, d_in, numElements);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppReduce()\n");
            abort();
        }
    }

    ~CUDPP_Reduce() {
        CUDPPResult result;
        result = cudppDestroyPlan(plan);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppDestroyPlan()\n");
        }
        result = cudppDestroy(theCudpp);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error cudppDestroy()\n");
        }
    }
};

/* CUDPP Hash*/
struct CUDPP_Compacting_Hash   {
    CUDPPHandle theCudpp;
    CUDPPHashTableConfig config;
    CUDPPHandle hash_table_handle;
    Int compacting_size = 0;

    void initialize(Int size)   {
        // Initialize CUDPP library
        CUDPPResult result = cudppCreate(&theCudpp);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error initializing CUDPP Library.\n");
        }

        // Initialze hash table
        config.type = CUDPP_COMPACTING_HASH_TABLE;
        config.kInputSize = size;
        config.space_usage = 1.5f;
        result = cudppHashTable(theCudpp, &hash_table_handle, &config);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_COMPACTING_HASH_TABLE cudppHashTable()\n");
        }
    }

    void insert(uint32_t *d_keys, Int length)   {
        uint32_t *d_vals = NULL;
        CUDPPResult result = cudppHashInsert(hash_table_handle, d_keys, d_vals, length);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Compacting_Hash cudppHashInsert()\n");
            abort();
        }
        result = cudppCompactingHashGetSize(hash_table_handle, (uint32_t*) &compacting_size);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Compacting_Hash cudppCompactingHashGetSize()\n");
            abort();
        }
    }

    void retrieve(uint32_t *d_keys, uint32_t *d_results, Int length)   {
        CUDPPResult result = cudppHashRetrieve(hash_table_handle, d_keys, d_results, length);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Compacting_Hash cudppHashRetrieve.\n");
        }
    }

    void destroy()  {
        CUDPPResult result = cudppDestroyHashTable(theCudpp, hash_table_handle);
        if (result != CUDPP_SUCCESS)  {
            fprintf(stderr, "Error in cudppHashTable destruction\n");
        }
        result = cudppDestroy(theCudpp);
        if (result != CUDPP_SUCCESS)  {
            fprintf(stderr, "Error in cudppDestroy\n");
        }
    }
};

struct CUDPP_Basic_Hash  {
    CUDPPHandle theCudpp;
    CUDPPHashTableConfig config;
    CUDPPHandle hash_table_handle;
    
    void initialize(Int size)   {
        // Initialize CUDPP library
        CUDPPResult result = cudppCreate(&theCudpp);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error initializing CUDPP Library.\n");
        }

        // Initialze hash table
        config.type = CUDPP_BASIC_HASH_TABLE;
        config.kInputSize = size;
        config.space_usage = 4.0f;
        result = cudppHashTable(theCudpp, &hash_table_handle, &config);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Basic_Hash cudppHashTable()\n");
        }
    }

    void destroy()  {
        CUDPPResult result = cudppDestroyHashTable(theCudpp, hash_table_handle);
        if (result != CUDPP_SUCCESS)  {
            fprintf(stderr, "Error in cudppHashTable destruction\n");
        }
        result = cudppDestroy(theCudpp);
        if (result != CUDPP_SUCCESS)  {
            fprintf(stderr, "Error in cudppDestroy\n");
        }
    }

    void insert(uint32_t *d_keys, uint32_t *d_vals, Int length)   {
        CUDPPResult result = cudppHashInsert(hash_table_handle, d_keys, d_vals, length);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Basic_Hash cudppHashInsert()\n");
            abort();
        }
        // cudaThreadSynchronize();
    }

    void retrieve(uint32_t *d_keys, uint32_t *d_results, Int length)   {
        CUDPPResult result = cudppHashRetrieve(hash_table_handle, d_keys, d_results, length);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Basic_Hash cudppHashRetrieve\n");
        }
    }

};


struct CUDPP_Multival_Hash  {
    CUDPPHandle theCudpp;
    CUDPPHashTableConfig config;
    CUDPPHandle hash_table_handle;
    
    uint32_t all_values_size; // input points num
    uint32_t index_counts_size; // output(unique) points num
    uint32_t *d_all_values;     // input points index, size = all_values_size
    uint2 *d_index_counts;   // x : first postion of output in table val array, y : number of same inputs of unique output, size = index_counts_size


    void initialize(Int size)   {
        // Initialize CUDPP library
        CUDPPResult result = cudppCreate(&theCudpp);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error initializing CUDPP Library.\n");
        }

        // Initialze hash table
        config.type = CUDPP_MULTIVALUE_HASH_TABLE;
        config.kInputSize = size;
        config.space_usage = 4.0f;
        result = cudppHashTable(theCudpp, &hash_table_handle, &config);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Multival_Hash cudppHashTable()\n");
        }
    }

    void destroy()  {
        CUDPPResult result = cudppDestroyHashTable(theCudpp, hash_table_handle);
        if (result != CUDPP_SUCCESS)  {
            fprintf(stderr, "Error in cudppHashTable destruction\n");
        }
        result = cudppDestroy(theCudpp);
        if (result != CUDPP_SUCCESS)  {
            fprintf(stderr, "Error in cudppDestroy\n");
        }
    }

    void insert(uint32_t *d_keys, uint32_t *d_vals, Int length)   {
        CUDPPResult result = cudppHashInsert(hash_table_handle, d_keys, d_vals, length);
        if (result != CUDPP_SUCCESS) {
            fprintf(stderr, "Error CUDPP_Multival_Hash cudppHashInsert()\n");
            abort();
        }
        if (cudppMultivalueHashGetValuesSize(hash_table_handle, &all_values_size) != CUDPP_SUCCESS) { 
            fprintf(stderr, "Error: cudppMultivalueHashGetValuesSize()\n");
            abort();
        }
        if (cudppMultivalueHashGetAllValues(hash_table_handle,  &d_all_values) !=  CUDPP_SUCCESS) {
            fprintf(stderr, "Error: cudppMultivalueHashGetAllValues()\n");
            abort();
        }
        if (cudppMultivalueHashGetIndexCounts(hash_table_handle, &d_index_counts, &index_counts_size) !=  CUDPP_SUCCESS) {
            fprintf(stderr, "Error: cudppMultivalueHashGetIndexCounts()\n");
            abort();
        }
    }
    
    uint32_t get_values_size()   {
        uint32_t values_size;
        if (cudppMultivalueHashGetValuesSize(hash_table_handle, &values_size) != CUDPP_SUCCESS) { 
          fprintf(stderr, "Error: cudppMultivalueHashGetValuesSize()\n");
          abort();
        }
        return values_size;
    }
    
    uint32_t* get_all_values()  {
        uint32_t* d_all_values = NULL;
        if (cudppMultivalueHashGetAllValues(hash_table_handle,  &d_all_values) !=  CUDPP_SUCCESS) {
            fprintf(stderr, "Error: cudppMultivalueHashGetAllValues()\n");
        }
        return d_all_values;
    }
    // multival hash retrieve will map key to (first val location, val num), value belong to same key will be continuously stored.
    void retrieve(uint32_t *d_query_keys, uint2 *d_query_vals_multivalue, Int query_size)   {
        CUDPPResult result = cudppHashRetrieve(hash_table_handle,  d_query_keys,  d_query_vals_multivalue,  query_size);
        if (result != CUDPP_SUCCESS)  { 
          fprintf(stderr, "Error: CUDPP_Multival_Hash cudppHashRetrieve()\n");
        }
    }
};


/* Basic Point Hash */
void dGetPointHash(uint32_t* d_points, uint32_t* d_keys, uint32_t* d_index, Int num);
void dGetPointHashQuery(uint32_t* d_points, uint32_t* d_keys, Int num);

void dMultivalHashGetValue(Int* _d_hash_idx, 
                            uint32_t* d_all_values, 
                            uint2* d_query_vals_multivalue,
                            uint32_t* d_results,
                            Int query_size,
                            Int hashtable_size);

template <Int Dimension>
class Basic_Point_Hashtable {
public:
    Int limit_size;
    Int size;
    CUDPP_Multival_Hash main_hash;
    
    /*Device Memory*/
    Int* _d_hash_points = NULL;
    Int* _d_hash_vals = NULL;


    Basic_Point_Hashtable(Int hashtable_size) : limit_size(hashtable_size), size(0)
    {
        main_hash.initialize(hashtable_size);
        gpuErrchk(cudaMalloc((void**)&_d_hash_points, sizeof(Int) * hashtable_size * Dimension));
        gpuErrchk(cudaMalloc((void**)&_d_hash_vals, sizeof(uint32_t) * hashtable_size));

    }
    
    ~Basic_Point_Hashtable()    {
        main_hash.destroy();
        gpuErrchk(cudaFree(_d_hash_points));
        gpuErrchk(cudaFree(_d_hash_vals));
    }
    
    void insert (uint32_t* d_points, uint32_t* d_vals, Int num)    {
        /* 
         * 1. hash
         * 2. insert
         * 3. copy points & vals
         **/
        uint32_t *d_keys = NULL;  /* query keys*/
        uint32_t *d_index = NULL;  /* query keys*/
        // Allocate memory for keys
        gpuErrchk(cudaMalloc((void**)&d_keys, sizeof(uint32_t) * num));
        gpuErrchk(cudaMalloc((void**)&d_index, sizeof(uint32_t) * num));

        dGetPointHash(d_points, d_keys, d_index, num);

        // Debug: check keys
        // uint32_t *h_keys = new uint32_t[num];
        // gpuErrchk(cudaMemcpy(h_keys, d_keys, sizeof(uint32_t) * num, cudaMemcpyDeviceToHost));
        // for(Int i = 0; i < num; i++)    {
        //     printf("0x%08x\n", h_keys[i]);
        // }
        // delete [] h_keys;

        main_hash.insert(d_keys, d_index, num);

        gpuErrchk(cudaMemcpy(_d_hash_points, d_points, sizeof(Int) * num * Dimension, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(_d_hash_vals, d_vals, sizeof(uint32_t) * num, cudaMemcpyDeviceToDevice));

        gpuErrchk(cudaFree(d_keys));
        gpuErrchk(cudaFree(d_index));

        size += num;
    }
    
    void retrieve (uint32_t* d_points, uint32_t* d_results, Int num)    {
        /*
         * 1. points to keys
         * 2. retrieve
         **/
        uint32_t *d_query_keys = NULL;  /* query keys*/
        // Allocate memory for keys
        gpuErrchk(cudaMalloc((void**)&d_query_keys, sizeof(uint32_t) * num));
        dGetPointHashQuery(d_points, d_query_keys, num);

        uint32_t *d_all_values = main_hash.get_all_values();

        uint2 *d_query_vals_multivalue = NULL;
        gpuErrchk(cudaMalloc((void**)&d_query_vals_multivalue, sizeof(uint2) * num));
        
        main_hash.retrieve(d_query_keys, d_query_vals_multivalue, num);
        
        // abort();

        dMultivalHashGetValue(_d_hash_vals, 
                              d_all_values, 
                              d_query_vals_multivalue,
                              d_results,
                              num,
                              size);

//        gpuErrchk(cudaFree(d_query_vals_multivalue));
        gpuErrchk(cudaFree(d_query_keys));
    }
};

/* Compacting Point Hash*/
void dCopyUniquePoints(uint32_t* _d_hash_points, uint32_t* d_points, uint32_t* d_index, Int num, Int size);

template <Int Dimension>
class Compacting_Point_Hashtable {
public:
    Int size;
    CUDPP_Compacting_Hash main_hash;
    /* Device Memory */
    Int* _d_hash_points = NULL;

    Compacting_Point_Hashtable() : size(-1)   {}
    
    ~Compacting_Point_Hashtable()    {
        // printf("Compacting_Point_Hashtable Destroy...\n");
        if(size != -1)  {
            main_hash.destroy();
        }
        if(_d_hash_points != NULL)  {
            gpuErrchk(cudaFree(_d_hash_points));
        }
    }
    
    void insert (uint32_t* d_points, Int num)    {
        /*
           d_points (0-num:x, num-2num:y, 2num-3num:z)
         * 1. compacting
         * 2. retrieve
         * 3. copy
         * 4, return max stash size
         **/
        assert(size == -1);

        main_hash.initialize(num);
        uint32_t *d_keys = NULL;  /* query keys*/
        uint32_t *d_index = NULL;  /* query keys*/
        gpuErrchk(cudaMalloc((void**)&d_keys, sizeof(uint32_t) * num));
        gpuErrchk(cudaMalloc((void**)&d_index, sizeof(uint32_t) * num));
        dGetPointHashQuery(d_points, d_keys, num);
 
        main_hash.insert(d_keys, num);
        size = main_hash.compacting_size;

        gpuErrchk(cudaMalloc((void**)&_d_hash_points, sizeof(Int) * size * Dimension));
        // d_index is unique points location(0-size-1) of origin points array(0-num-1).
        main_hash.retrieve(d_keys, d_index, num);
    
        //save unique points to _d_hash_points.
        dCopyUniquePoints((uint32_t*)_d_hash_points, d_points, d_index, num, size);
        gpuErrchk(cudaFree(d_keys));
        gpuErrchk(cudaFree(d_index));

        // Debug
        // Int* h_result = new Int[size * Dimension];
        // gpuErrchk(cudaMemcpy(h_result, _d_hash_points, sizeof(Int) * size * Dimension, cudaMemcpyDeviceToHost));
        // printf("_d_hash_points:\n");
        // for(Int i = 0; i < size; i++) {
        //     for(Int j = 0; j < Dimension; j++) {
        //         printf("%d\t", h_result[i + j * size]);
        //     }
        //     printf("\n");
        // }
        // delete [] h_result;
    }

    void retrieve (uint32_t* d_points, uint32_t* d_results, Int num)    {
        /*
         * 1. points to keys
         * 2. retrieve
         **/

        assert(size != -1);
        
        uint32_t *d_keys = NULL;  /* query keys*/
        // Allocate memory for keys
        gpuErrchk(cudaMalloc((void**)&d_keys, sizeof(uint32_t) * num));

        dGetPointHashQuery(d_points, d_keys, num);

        main_hash.retrieve(d_keys, d_results, num);

        gpuErrchk(cudaFree(d_keys));
    }

    Int getCompactingSize() {
        return size;
    }

    Int* getAllPoints() {
        return _d_hash_points;
    }

    void insert_points(Points<Dimension> &points)    {
        Int num = points.size();
        Int *points_int = new Int[num * Dimension];
        for(Int i = 0; i < (Int)num; i++)  {
            for(Int d = 0; d < Dimension; d++)  {
                points_int[d * num + i] = points[i][d];
            }
        }
        uint32_t *d_points = NULL;

        // Copy points to GPU
        gpuErrchk(cudaMalloc((void**)&d_points, sizeof(uint32_t) * num * Dimension));
        gpuErrchk(cudaMemcpy(d_points, points_int, sizeof(uint32_t) * num * Dimension, cudaMemcpyHostToDevice));
        insert(d_points, num);
        gpuErrchk(cudaFree(d_points));
        delete [] points_int;
    }

    void retrieve_points (Points<Dimension> &points, Ints &vals)    {
        Int num = points.size();
        Int *points_int = new Int[num * Dimension];
        for(Int i = 0; i < (Int)num; i++)  {
            for(Int d = 0; d < Dimension; d++)  {
                points_int[d * num + i] = points[i][d];
            }
        }
        uint32_t *d_points = NULL;
        uint32_t *d_results = NULL;  /* query results*/
        // Allocate memory for results 
        gpuErrchk(cudaMalloc((void**)&d_results, sizeof(uint32_t) * num));
        // Copy points to GPU
        gpuErrchk(cudaMalloc((void**)&d_points, sizeof(uint32_t) * num * Dimension));
        gpuErrchk(cudaMemcpy(d_points, points_int, sizeof(uint32_t) * num * Dimension, cudaMemcpyHostToDevice));

        retrieve (d_points, d_results, num);

        vals.resize(num);
        gpuErrchk(cudaMemcpy(vals.data(), d_results, sizeof(uint32_t) * num, cudaMemcpyDeviceToHost));                

        gpuErrchk(cudaFree(d_points));
        gpuErrchk(cudaFree(d_results));
        delete [] points_int;
    }
};



void dMultivalPointHashtableInsertHelper(
        uint2* d_index_counts,
        uint32_t* d_all_values,
        uint32_t* d_points,
        Int* _d_hash_idx,
        Int* _d_hash_points,
        uint32_t index_counts_size,
        uint32_t all_values_size);


void PreprocessPointClouds(long *d_ori_points , uint32_t * d_points, uint32_t* d_keys, uint32_t* d_index,
                           uint32_t *d_point_cloud_start , const Int batch_size, const int nInputRows);
void GetRepeatNums(uint2 *d_index_counts, uint32_t* d_repeatNums, int size);
void CollectInputRules(uint2 * d_index_counts,uint32_t * d_all_values, Int * d_input_rules, int size, int max_repeat_num);

template <Int Dimension>
void HashInputPointClouds(long* coords_ptr, uint32_t *point_cloud_start,
                          uint32_t * d_points, uint32_t* d_keys, uint32_t* d_index,
                          const Int nInputRows, const Int batch_size){


    long *d_ori_points = NULL;
    uint32_t *d_point_cloud_start = NULL;
    gpuErrchk(cudaMalloc((void**)&d_ori_points, sizeof(long) * nInputRows * (Dimension + 1)));
    gpuErrchk(cudaMalloc((void**)&d_point_cloud_start, sizeof(uint32_t) * (batch_size + 1)));
    gpuErrchk(cudaMemcpy(d_ori_points, coords_ptr, sizeof(long) * nInputRows * (Dimension + 1), cudaMemcpyHostToDevice));
    PreprocessPointClouds(d_ori_points, d_points, d_keys, d_index, d_point_cloud_start, batch_size, nInputRows);
    gpuErrchk(cudaMemcpy(point_cloud_start, d_point_cloud_start, sizeof(uint32_t) * (batch_size + 1), cudaMemcpyDeviceToHost));
    point_cloud_start[0] = 0;
    point_cloud_start[batch_size] = nInputRows;
    gpuErrchk(cudaFree(d_ori_points));
    gpuErrchk(cudaFree(d_point_cloud_start));

}
inline void getMaximumValue(uint32_t * d_input, int num_elements,int &max_value)
{

    // Initialize CUDPP
    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error initializing CUDPP Library\n");
        return ;
    }

    CUDPPConfiguration config;
    config.algorithm = CUDPP_REDUCE;
    config.options = 0;
    config.op = CUDPP_MAX;
    config.datatype = CUDPP_INT;
    CUDPPHandle plan;

    result = cudppPlan(theCudpp, &plan, config, num_elements, 1, 0);

    if(result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation\n");
        cudppDestroyPlan(plan);
        cudppDestroy(theCudpp);
        return ;
    }
    uint32_t * d_output = NULL;
    gpuErrchk(cudaMalloc((void**)&d_output, sizeof(uint32_t) * 1));
    cudppReduce(plan, d_output, d_input, num_elements);
    gpuErrchk(cudaMemcpy(&max_value, d_output, sizeof(uint32_t) * 1, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_output));


    result = cudppDestroyPlan(plan);
    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying CUDPPPlan for Scan\n");
    }
    result = cudppDestroy(theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        printf("Error shutting down CUDPP Library.\n");
    }

}

template <Int Dimension>
class Multival_Point_Hashtable {
public:
    Int size;//unique key size
    CUDPP_Multival_Hash main_hash;
    
    /*Device Memory*/
    Int* _d_hash_points = NULL; // output points coords, size = main_hash.index_counts_size*3
    Int* _d_hash_idx = NULL;    // map input location to output location, size = main_hash.all_values_size

    Multival_Point_Hashtable() : size(-1)   {}
    
    ~Multival_Point_Hashtable()    {
        if(size != -1)  {
            main_hash.destroy();
            gpuErrchk(cudaFree(_d_hash_points));
            gpuErrchk(cudaFree(_d_hash_idx));
        }
    }
    void generateInputRules(Int * rules, int max_repeat_num)
    {
        assert(size > 0);
        Int  * d_rules = NULL;
        gpuErrchk(cudaMalloc((void**)&d_rules , sizeof(Int ) * (max_repeat_num + 1) * size));
        // map output location to input location, size= outputsize * (max_repeat_num+1)
        CollectInputRules(main_hash.d_index_counts, main_hash.d_all_values, d_rules,size, max_repeat_num);
        gpuErrchk(cudaMemcpy(rules, d_rules, sizeof(Int) * size * (max_repeat_num + 1), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(d_rules));
    }

    void InsertAndCompactPointCloud(uint32_t* d_keys, uint32_t* d_index, uint32_t * d_points,
                                    Int num, Int all_values_size, int &max_repeat_num){
        assert(size == -1);
        main_hash.initialize(num);
        main_hash.insert(d_keys, d_index, num);
        // unique key size 
        size = main_hash.index_counts_size;

        uint32_t *dRepeatNums = NULL;
        gpuErrchk(cudaMalloc((void**)&dRepeatNums , sizeof(uint32_t) * size));
        // get repeat time of each key from d_index_counts.y to dRepeatNums
        GetRepeatNums(main_hash.d_index_counts, dRepeatNums, size);
        // get array max val.
        getMaximumValue(dRepeatNums,size,max_repeat_num);
        gpuErrchk(cudaFree(dRepeatNums));

        // d_index_counts -> idx, d_points -> _d_hash_points
        assert(num == (Int)main_hash.all_values_size);

        gpuErrchk(cudaMalloc((void**)&_d_hash_idx, sizeof(Int) * all_values_size));
        gpuErrchk(cudaMalloc((void**)&_d_hash_points, sizeof(Int) * size * Dimension));
        // get multival hashtable _d_hash_idx(size=all_values_size, val=first index in val table), _d_hash_points(3*index_counts_size, points)
        dMultivalPointHashtableInsertHelper(
            main_hash.d_index_counts,
            main_hash.d_all_values,
            d_points,
            _d_hash_idx,
            _d_hash_points,
            main_hash.index_counts_size,
            all_values_size);
    }

    void insert (uint32_t* d_points, Int num)    {
        /* 
         * 1. hash
         * 2. insert
         * 3. copy points & vals
         **/



        EASY_BLOCK("instrict retrieve"); 
        assert(size == -1);

        main_hash.initialize(num);

        uint32_t *d_keys = NULL;  /* query keys*/
        uint32_t *d_index = NULL;  /* query keys*/
        // Allocate memory for keys
        gpuErrchk(cudaMalloc((void**)&d_keys, sizeof(uint32_t) * num));
        gpuErrchk(cudaMalloc((void**)&d_index, sizeof(uint32_t) * num));

        // get points hash and index(0-num-1)
        dGetPointHash(d_points, d_keys, d_index, num);
        main_hash.insert(d_keys, d_index, num);

        size = main_hash.index_counts_size;

        // d_index_counts -> idx, d_points -> _d_hash_points
        assert(num == (Int)main_hash.all_values_size);
        gpuErrchk(cudaMalloc((void**)&_d_hash_idx, sizeof(Int) * num));
        gpuErrchk(cudaMalloc((void**)&_d_hash_points, sizeof(Int) * size * Dimension));

        // generate _d_hash_idx and _d_hash_points.
        dMultivalPointHashtableInsertHelper(
            main_hash.d_index_counts, 
            main_hash.d_all_values,
            d_points,
            _d_hash_idx,
            _d_hash_points,
            main_hash.index_counts_size,
            main_hash.all_values_size);

        gpuErrchk(cudaFree(d_keys));
        gpuErrchk(cudaFree(d_index));

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;
    }
    
    void retrieve (uint32_t* d_points, uint32_t* d_results, Int num)    {
        /*
         * 1. @input
             d_points:      input points 
             num:           input points num
         * 2. @output 
             d_results:     output location (0-unique num)
         **/
        EASY_FUNCTION(profiler::colors::Brown200);

        EASY_BLOCK("get key"); 
        assert(size != -1);

        uint32_t *d_query_keys = NULL;  /* query keys*/
        // Allocate memory for keys
        gpuErrchk(cudaMalloc((void**)&d_query_keys, sizeof(uint32_t) * num));
        dGetPointHashQuery(d_points, d_query_keys, num);


        // Debug: check keys
        // printf("retrieve keys\n");
        // uint32_t *h_keys = new uint32_t[num];
        // gpuErrchk(cudaMemcpy(h_keys, d_query_keys, sizeof(uint32_t) * num, cudaMemcpyDeviceToHost));
        // for(Int i = 0; i < num; i++)    {
        //     printf("0x%08x\n", h_keys[i]);
        // }
        // delete [] h_keys;


        uint32_t *d_all_values = main_hash.get_all_values();

        uint2 *d_query_vals_multivalue = NULL;
        gpuErrchk(cudaMalloc((void**)&d_query_vals_multivalue, sizeof(uint2) * num));
        
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;

        EASY_BLOCK("retrieve");
        
        main_hash.retrieve(d_query_keys, d_query_vals_multivalue, num);
        
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;

        EASY_BLOCK("dMultivalHashGetValue"); 
        // use retreive result(uint2) to get input first location of the points. no exist return -1.
        dMultivalHashGetValue(_d_hash_idx, 
                              d_all_values, 
                              d_query_vals_multivalue,
                              d_results,
                              num,
                              size);

        gpuErrchk(cudaFree(d_query_vals_multivalue));
        gpuErrchk(cudaFree(d_query_keys));

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;
    }

    Int getCompactingSize() {
        return size;
    }

    Int* getAllPoints() {
        return _d_hash_points;
    }


    void insert_points(Points<Dimension> &points)    {
        Int num = points.size();
        Int *points_int = new Int[num * Dimension];
        for(Int i = 0; i < (Int)num; i++)  {
            for(Int d = 0; d < Dimension; d++)  {
                points_int[d * num + i] = points[i][d];
            }
        }
        uint32_t *d_points = NULL;

        // Copy points to GPU
        gpuErrchk(cudaMalloc((void**)&d_points, sizeof(uint32_t) * num * Dimension));
        gpuErrchk(cudaMemcpy(d_points, points_int, sizeof(uint32_t) * num * Dimension, cudaMemcpyHostToDevice));
        insert(d_points, num);
        gpuErrchk(cudaFree(d_points));
        delete [] points_int;
    }

    void retrieve_points (Points<Dimension> &points, Ints &vals)    {
        Int num = points.size();
        Int *points_int = new Int[num * Dimension];
        for(Int i = 0; i < (Int)num; i++)  {
            for(Int d = 0; d < Dimension; d++)  {
                points_int[d * num + i] = points[i][d];
            }
        }
        uint32_t *d_points = NULL;
        uint32_t *d_results = NULL;  /* query results*/
        // Allocate memory for results 
        gpuErrchk(cudaMalloc((void**)&d_results, sizeof(uint32_t) * num));
        // Copy points to GPU
        gpuErrchk(cudaMalloc((void**)&d_points, sizeof(uint32_t) * num * Dimension));
        gpuErrchk(cudaMemcpy(d_points, points_int, sizeof(uint32_t) * num * Dimension, cudaMemcpyHostToDevice));

        retrieve (d_points, d_results, num);

        vals.resize(num);
        gpuErrchk(cudaMemcpy(vals.data(), d_results, sizeof(uint32_t) * num, cudaMemcpyDeviceToHost));                

        gpuErrchk(cudaFree(d_points));
        gpuErrchk(cudaFree(d_results));
        delete [] points_int;
    }

void retrieve_key (uint32_t* d_query_keys, uint32_t* d_results, Int num)    {
        /*
         * 1. input--d_query_keys: query keys
         * 2. output--d_results: output location coorresbond to input key
         **/
        EASY_FUNCTION(profiler::colors::Brown200);

        EASY_BLOCK("get key"); 
        assert(size != -1);

        uint32_t *d_all_values = main_hash.get_all_values();

        uint2 *d_query_vals_multivalue = NULL;
        gpuErrchk(cudaMalloc((void**)&d_query_vals_multivalue, sizeof(uint2) * num));
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;

        EASY_BLOCK("retrieve"); 
        main_hash.retrieve(d_query_keys, d_query_vals_multivalue, num);
        
        gpuErrchk(cudaDeviceSynchronize());

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;

        EASY_BLOCK("dMultivalHashGetValue"); 
        dMultivalHashGetValue(_d_hash_idx, 
                              d_all_values, 
                              d_query_vals_multivalue,
                              d_results,
                              num,
                              size);

        gpuErrchk(cudaFree(d_query_vals_multivalue));
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;
    }
};

#endif
