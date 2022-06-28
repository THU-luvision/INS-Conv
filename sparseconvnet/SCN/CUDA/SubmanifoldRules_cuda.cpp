// Added by Tian Zheng
// t-zheng@outlook.com

#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "CUDPPWrapper.hpp"
#include "../Metadata/Metadata.h"
#include "../Metadata/RectangularRegions.h"
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/types.h>
#include <unistd.h>
void dGetPointsWithinInputRegion(const Int *d_points, Int *d_output_points, Int num, Int size, Int Dimension);

void dGetNot(Int *a, Int num);

void dGenerateRulebook (Int* d_global_query_result,    // size = NActivePoints * volume
	Int* d_out_rules,              // size = volume * (NActivePoints * 2)
	unsigned* d_out_rules_flag,    // size = volume * (NActivePoints * 2)
	Int NActivePoints,
	Int volume,
    Int ctr);

void dGetChunkPointsNum(Int *chunk_ids, Int *chunk_num, Int points_num);

void dGenerateIncreRulebook (Int* d_global_query_result,    // size = NActivePoints * volume
                        Int *d_output_query_result,
                        Int* d_out_rules,              // size = volume * (NActivePoints * 2)
                        unsigned* d_out_rules_flag,    // size = volume * (NActivePoints * 2)
                        Int NActivePoints,
                        Int volume,
                        Int ctr);
template <Int Dimension>
class SubmanifoldBuildObject
{
    public:
    Int NActivePoints;
    Int conv_size;
    Int volume;
    Int ctr;
    /* Device Memory */
    Int *d_global_points = NULL; /**/
    // Int *d_global_addresses = NULL;
    Int *d_global_query_list = NULL;     /* NActivePoints * volume */
    Int *d_global_query_result = NULL;   /*NActivePoints * volume*/
    
    /* High-level structure*/
    // Basic_Point_Hashtable<Dimension> global_hashtable;
    GPU_SparseGrid<Dimension> &SG;

    SubmanifoldBuildObject(Int NActivePoints_, long* conv_size_, GPU_SparseGrid<Dimension> &SG_) : SG(SG_)
    {
        if (Dimension != 3)
        {
            fprintf(stderr, "dimension != 3\n");
            abort();
        }
        NActivePoints = NActivePoints_;
        conv_size = (Int)conv_size_[0];
        assert(conv_size == 3);
        volume = 1;
        for(Int i = 0; i < Dimension; i++) {
          volume *= (Int)conv_size_[i];
        }

        d_global_points = SG.pHash->getAllPoints();
        ctr = SG.ctr;

        initialize(NActivePoints_);
    }

    void initialize(Int length)
    {
        // h_global_points = new Int[length * Dimension];
        // h_global_addresses = new Int[length];
        // gpuErrchk(cudaMalloc((void **)&d_global_points, sizeof(Int) * NActivePoints * Dimension));
        // gpuErrchk(cudaMalloc((void **)&d_global_addresses, sizeof(Int) * NActivePoints));
        gpuErrchk(cudaMalloc((void **)&d_global_query_list, sizeof(Int) * NActivePoints * volume));
        gpuErrchk(cudaMalloc((void **)&d_global_query_result, sizeof(Int) * NActivePoints * volume));
    }

    ~SubmanifoldBuildObject()
    {
        // if (h_global_points != NULL)
        // {
        //     delete[] h_global_points;
        // }
        // if (h_global_addresses != NULL)
        // {
        //     delete[] h_global_addresses;
        // }
        // if (d_global_points != NULL)
        // {
        //     gpuErrchk(cudaFree(d_global_points));
        // }
        // if (d_global_addresses != NULL)
        // {
        //     gpuErrchk(cudaFree(d_global_addresses));
        // }
        if (d_global_query_list != NULL)
        {
            gpuErrchk(cudaFree(d_global_query_list));
        }
        if (d_global_query_result != NULL)
        {
            gpuErrchk(cudaFree(d_global_query_result));
        }
    }
    
    /*
        get input region of output
        @input:
            d_global_points: output points
        @output:
            d_global_query_list: input regions hash key.
    */
    void generateQueryList()
    {
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_FUNCTION(profiler::colors::Amber200);
        // Generate query list on GPU
        dGetPointsWithinInputRegion(d_global_points, d_global_query_list, NActivePoints, conv_size, Dimension);
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
    }

    // void copy_from_host_to_device()
    // {
    //     // Memory Copy
    //     gpuErrchk(cudaMemcpy(d_global_points, h_global_points, sizeof(Int) * NActivePoints * Dimension, cudaMemcpyHostToDevice));
    //     gpuErrchk(cudaMemcpy(d_global_addresses, h_global_addresses, sizeof(Int) * NActivePoints, cudaMemcpyHostToDevice));
    // }
    
    // void global_hash_insert()
    // {
    //     global_hashtable.insert((uint32_t *)d_global_points, (uint32_t *)d_global_addresses, NActivePoints);
    // }

    void global_hash_query()
    {
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_FUNCTION(profiler::colors::Amber200);
        // global_hashtable.retrieve((uint32_t *)d_global_query_list, (uint32_t *)d_global_query_result, NActivePoints * volume);
        SG.pHash->retrieve_key((uint32_t *)d_global_query_list, (uint32_t *)d_global_query_result, NActivePoints * volume);
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
    }


    Int generate_rulebook(RuleBook &rules)
    {

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_FUNCTION(profiler::colors::Amber200);
        /* Device Memory */
        Int *d_out_rules = NULL;
        Int *d_out_rules_tmp = NULL;
        uint32_t *d_out_rules_flag = NULL;
        uint32_t *d_out_rules_flag_tmp = NULL;
        Int *d_out_rules_compacted = NULL;
        size_t *d_numValid = NULL;

        gpuErrchk(cudaMalloc((void **)&d_out_rules, sizeof(Int) * volume * (NActivePoints * 2)));
        gpuErrchk(cudaMalloc((void **)&d_out_rules_tmp, sizeof(Int) * (NActivePoints * 2)));
        gpuErrchk(cudaMalloc((void **)&d_out_rules_flag, sizeof(uint32_t) * volume * (NActivePoints * 2)));
        gpuErrchk(cudaMemset(d_out_rules_flag, 0, sizeof(uint32_t) * volume * (NActivePoints * 2)));
        gpuErrchk(cudaMalloc((void **)&d_out_rules_flag_tmp, sizeof(uint32_t) * (NActivePoints * 2)));

        gpuErrchk(cudaMalloc((void **)&d_out_rules_compacted, sizeof(Int) * volume * (NActivePoints * 2)));
        gpuErrchk(cudaMalloc((void **)&d_numValid, sizeof(size_t) * volume));

        dGenerateRulebook (d_global_query_result,    // size = NActivePoints * volume
                           d_out_rules,              // size = volume * (NActivePoints * 2)
                           d_out_rules_flag,    // size = volume * (NActivePoints * 2)
                           NActivePoints,
                           volume,
                           ctr);

        CUDPP_Compacting compacting(NActivePoints * 2);
        for(Int i = 0; i < volume; i++) {
            gpuErrchk(cudaMemcpy(d_out_rules_tmp, &d_out_rules[i * (NActivePoints * 2)], sizeof(Int) * (NActivePoints * 2), cudaMemcpyDeviceToDevice));
            gpuErrchk(cudaMemcpy(d_out_rules_flag_tmp, &d_out_rules_flag[i * (NActivePoints * 2)], sizeof(uint32_t) * (NActivePoints * 2), cudaMemcpyDeviceToDevice));
            compacting.apply(&d_out_rules_compacted[i * (NActivePoints * 2)], 
                             &d_numValid[i],
                             d_out_rules_tmp,
                             d_out_rules_flag_tmp,
                             (size_t)NActivePoints * 2);
        }

        size_t *h_numValid = new size_t[volume];
        gpuErrchk(cudaMemcpy(h_numValid, d_numValid, sizeof(size_t) * volume, cudaMemcpyDeviceToHost));

        Int countActiveInputs = 0;
        for(Int i = 0; i < volume; i++) {
            Int old_size = rules[i].size();
            rules[i].resize(old_size + h_numValid[i]);
            gpuErrchk(cudaMemcpy(rules[i].data() + old_size, &d_out_rules_compacted[i * (NActivePoints * 2)], sizeof(Int) * (Int)h_numValid[i], cudaMemcpyDeviceToHost));
            countActiveInputs += (Int)h_numValid[i] / 2;
        }

        gpuErrchk(cudaFree(d_out_rules));
        gpuErrchk(cudaFree(d_out_rules_tmp));
        gpuErrchk(cudaFree(d_out_rules_flag));
        gpuErrchk(cudaFree(d_out_rules_flag_tmp));
        gpuErrchk(cudaFree(d_out_rules_compacted));
        gpuErrchk(cudaFree(d_numValid));

        delete [] h_numValid;

        return countActiveInputs;
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
    }
};

/*********************************************************************************************************************************************************************************/


template <Int Dimension>
class IncreSubmanifoldBuildObject
{
    public:
    size_t NActivePoints;
    Int conv_size;
    Int volume;
    Int ctr;
    /* Device Memory */
    Int *d_global_points = NULL; /**/
    // Int *d_global_addresses = NULL;
    Int *d_global_query_list = NULL;     /* NActivePoints * volume */
    Int *d_global_query_result = NULL;   /*NActivePoints * volume*/
    Int *d_output_query_result = NULL;
    /* High-level structure*/
    // Basic_Point_Hashtable<Dimension> global_hashtable;
    GPU_SparseGrid<Dimension> &SG;
    GPU_SparseGrid<Dimension> &pre_SG;
    Int *d_reduced_points = NULL; 
    size_t *d_reduced_num = NULL;

    IncreSubmanifoldBuildObject(size_t NActivePoints_, long* conv_size_, GPU_SparseGrid<Dimension> &SG_, GPU_SparseGrid<Dimension> &pre_SG_) : SG(SG_), pre_SG(pre_SG_)
    {
        if (Dimension != 3)
        {
            fprintf(stderr, "dimension != 3\n");
            abort();
        }
        NActivePoints = NActivePoints_;
        conv_size = (Int)conv_size_[0];
        assert(conv_size == 3);
        volume = 1;
        for(Int i = 0; i < Dimension; i++) {
          volume *= (Int)conv_size_[i];
        }

        d_global_points = SG.pHash->getAllPoints();

        Int *d_pre_not_exist = NULL;
        gpuErrchk(cudaMalloc((void**)&d_pre_not_exist, sizeof(Int)*NActivePoints));
        pre_SG.pHash->retrieve((uint32_t*) d_global_points, (uint32_t*) d_pre_not_exist, NActivePoints);
        dGetNot(d_pre_not_exist, NActivePoints);
        uint32_t *d_pre_not_exist_long = NULL;
        gpuErrchk(cudaMalloc((void**)&d_pre_not_exist_long, sizeof(uint32_t)*NActivePoints * Dimension));
        gpuErrchk(cudaMemcpy(d_pre_not_exist_long, d_pre_not_exist, sizeof(uint32_t)*NActivePoints,cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_pre_not_exist_long + NActivePoints, d_pre_not_exist, sizeof(uint32_t)*NActivePoints, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_pre_not_exist_long + 2 * NActivePoints, d_pre_not_exist, sizeof(uint32_t)*NActivePoints, cudaMemcpyDeviceToDevice));
        CUDPP_Compacting compacting(NActivePoints * Dimension);
        gpuErrchk(cudaMalloc((void**)&d_reduced_points, sizeof(Int)*NActivePoints*Dimension));
        gpuErrchk(cudaMalloc((void**)&d_reduced_num, sizeof(size_t)));
        compacting.apply(d_reduced_points, d_reduced_num, d_global_points, d_pre_not_exist_long, NActivePoints*Dimension);
        d_global_points = d_reduced_points;

        gpuErrchk(cudaMemcpy(&NActivePoints, d_reduced_num, sizeof(size_t), cudaMemcpyDeviceToHost));
        NActivePoints /= 3;

        gpuErrchk(cudaFree(d_pre_not_exist));
        gpuErrchk(cudaFree(d_pre_not_exist_long));

        ctr = SG.ctr;
        initialize(NActivePoints);
    }

    void initialize(Int length)
    {
        // h_global_points = new Int[length * Dimension];
        // h_global_addresses = new Int[length];
        // gpuErrchk(cudaMalloc((void **)&d_global_points, sizeof(Int) * NActivePoints * Dimension));
        // gpuErrchk(cudaMalloc((void **)&d_global_addresses, sizeof(Int) * NActivePoints));
        gpuErrchk(cudaMalloc((void **)&d_global_query_list, sizeof(Int) * NActivePoints * volume));
        gpuErrchk(cudaMalloc((void **)&d_global_query_result, sizeof(Int) * NActivePoints * volume));
        gpuErrchk(cudaMalloc((void **)&d_output_query_result, sizeof(Int) * NActivePoints));
    }

    ~IncreSubmanifoldBuildObject()
    {
        // if (h_global_points != NULL)
        // {
        //     delete[] h_global_points;
        // }
        // if (h_global_addresses != NULL)
        // {
        //     delete[] h_global_addresses;
        // }
        // if (d_global_points != NULL)
        // {
        //     gpuErrchk(cudaFree(d_global_points));
        // }
        // if (d_global_addresses != NULL)
        // {
        //     gpuErrchk(cudaFree(d_global_addresses));
        // }
        if (d_global_query_list != NULL)
        {
            gpuErrchk(cudaFree(d_global_query_list));
        }
        if (d_global_query_result != NULL)
        {
            gpuErrchk(cudaFree(d_global_query_result));
        }
        if (d_reduced_points != NULL)
        {
            gpuErrchk(cudaFree(d_reduced_points));
        }    
        if (d_reduced_num != NULL)
        {
            gpuErrchk(cudaFree(d_reduced_num));
        }
        if (d_output_query_result != NULL) {
            gpuErrchk(cudaFree(d_output_query_result));
        }
    
    }
    
    /*
        get input region of output
        @input:
            d_global_points: output points
        @output:
            d_global_query_list: input regions hash key.
    */
    void generateQueryList()
    {
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_FUNCTION(profiler::colors::Amber200);
        // Generate query list on GPU
        if (NActivePoints == 0) {
            return;
        }
        dGetPointsWithinInputRegion(d_global_points, d_global_query_list, NActivePoints, conv_size, Dimension);
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
    }

    // void copy_from_host_to_device()
    // {
    //     // Memory Copy
    //     gpuErrchk(cudaMemcpy(d_global_points, h_global_points, sizeof(Int) * NActivePoints * Dimension, cudaMemcpyHostToDevice));
    //     gpuErrchk(cudaMemcpy(d_global_addresses, h_global_addresses, sizeof(Int) * NActivePoints, cudaMemcpyHostToDevice));
    // }
    
    // void global_hash_insert()
    // {
    //     global_hashtable.insert((uint32_t *)d_global_points, (uint32_t *)d_global_addresses, NActivePoints);
    // }

    void global_hash_query()
    {
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_FUNCTION(profiler::colors::Amber200);
        if (NActivePoints == 0) {
            return;
        }
        // global_hashtable.retrieve((uint32_t *)d_global_query_list, (uint32_t *)d_global_query_result, NActivePoints * volume);
        pre_SG.pHash->retrieve_key((uint32_t *)d_global_query_list, (uint32_t *)d_global_query_result, NActivePoints * volume);
        SG.pHash->retrieve((uint32_t *)d_global_points, (uint32_t*) d_output_query_result, NActivePoints);
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif  
    }


    Int generate_rulebook(RuleBook &rules)
    {

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_FUNCTION(profiler::colors::Amber200);
        /* Device Memory */
        if (NActivePoints == 0) {
            return 0;
        }


        Int *d_out_rules = NULL;
        Int *d_out_rules_tmp = NULL;
        uint32_t *d_out_rules_flag = NULL;
        uint32_t *d_out_rules_flag_tmp = NULL;
        Int *d_out_rules_compacted = NULL;
        size_t *d_numValid = NULL;

        gpuErrchk(cudaMalloc((void **)&d_out_rules, sizeof(Int) * volume * (NActivePoints * 2)));
        gpuErrchk(cudaMalloc((void **)&d_out_rules_tmp, sizeof(Int) * (NActivePoints * 2)));
        gpuErrchk(cudaMalloc((void **)&d_out_rules_flag, sizeof(uint32_t) * volume * (NActivePoints * 2)));
        gpuErrchk(cudaMemset(d_out_rules_flag, 0, sizeof(uint32_t) * volume * (NActivePoints * 2)));
        gpuErrchk(cudaMalloc((void **)&d_out_rules_flag_tmp, sizeof(uint32_t) * (NActivePoints * 2)));

        gpuErrchk(cudaMalloc((void **)&d_out_rules_compacted, sizeof(Int) * volume * (NActivePoints * 2)));
        gpuErrchk(cudaMalloc((void **)&d_numValid, sizeof(size_t) * volume));


        dGenerateIncreRulebook (d_global_query_result,    // size = NActivePoints * volume
                            d_output_query_result,
                           d_out_rules,              // size = volume * (NActivePoints * 2)
                           d_out_rules_flag,    // size = volume * (NActivePoints * 2)
                           NActivePoints,
                           volume,
                           ctr);
        CUDPP_Compacting compacting(NActivePoints * 2);
        for(Int i = 0; i < volume; i++) {
            gpuErrchk(cudaMemcpy(d_out_rules_tmp, &d_out_rules[i * (NActivePoints * 2)], sizeof(Int) * (NActivePoints * 2), cudaMemcpyDeviceToDevice));
            gpuErrchk(cudaMemcpy(d_out_rules_flag_tmp, &d_out_rules_flag[i * (NActivePoints * 2)], sizeof(uint32_t) * (NActivePoints * 2), cudaMemcpyDeviceToDevice));
            compacting.apply(&d_out_rules_compacted[i * (NActivePoints * 2)], 
                             &d_numValid[i],
                             d_out_rules_tmp,
                             d_out_rules_flag_tmp,
                             (size_t)NActivePoints * 2);
        }

        size_t *h_numValid = new size_t[volume];
        gpuErrchk(cudaMemcpy(h_numValid, d_numValid, sizeof(size_t) * volume, cudaMemcpyDeviceToHost));

        Int countActiveInputs = 0;
        for(Int i = 0; i < volume; i++) {
            rules[i].resize(h_numValid[i]);
            gpuErrchk(cudaMemcpy(rules[i].data(), &d_out_rules_compacted[i * (NActivePoints * 2)], sizeof(Int) * (Int)h_numValid[i], cudaMemcpyDeviceToHost));
            countActiveInputs += (Int)h_numValid[i] / 2;
        }

        gpuErrchk(cudaFree(d_out_rules));
        gpuErrchk(cudaFree(d_out_rules_tmp));
        gpuErrchk(cudaFree(d_out_rules_flag));
        gpuErrchk(cudaFree(d_out_rules_flag_tmp));
        gpuErrchk(cudaFree(d_out_rules_compacted));
        gpuErrchk(cudaFree(d_numValid));

        delete [] h_numValid;

        return countActiveInputs;
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
    }
};


/* Chuck based */ 
void dGetBlockGrid(Int *d_input_point, Int *d_blockid, Int block_size, Int length);

void dGetBlockAddressList(Int *d_input_point, 
                          Int *d_blockid, Int* d_input_address, 
                          Int* d_output_address, 
                          Int* d_input_count, 
                          Int* d_output_count);

void dGenerateChuckRulebook (Int* d_input_point_query_result,             // size = num_active_point * 27
                            Int* d_ori_index,                            // size = num_active_point
                            Int* d_active_output_point_blockid,          // size = num_active_point
                            Int* d_block_active_output_count,            /* initialize with 0 */

                            Int* d_block_input_address_list,
                            Int* d_block_input_address_list_size,

                            short* d_output_rulebook,                      // dim = (num_block, max_point_per_block * 27) 

                            Int* d_output_output_address,                // size = [num_block, max_point_per_block]

                            Int* d_output_input_address,

                            Int num_active_point,
                            Int num_block,
                            Int max_point_per_block,							
                            Int max_input_address,
                            Int ctr,
                            Int use_normal);

void dGenerateIncreChuckRulebook (Int* d_input_point_query_result,             // size = num_active_point * 27
                            Int* d_ori_index,                            // size = num_active_point
                            Int* d_output_query_result,
                            Int* d_active_output_point_blockid,          // size = num_active_point
                            Int* d_block_active_output_count,            /* initialize with 0 */

                            Int* d_block_input_address_list,
                            Int* d_block_input_address_list_size,
                            Int* d_chunk_points_num,
                            short* d_output_rulebook,                      // dim = (num_block, max_point_per_block * 27) 

                            Int* d_output_output_address,                // size = [num_block, max_point_per_block]

                            Int* d_output_input_address,

                            Int num_active_point,
                            Int num_block,
                            Int max_point_per_block,							
                            Int max_input_address,
                            Int ctr,
                            Int use_normal);


void dGenerateChunkBackwardRB(
    Int* d_input_point_query_result,             // size = num_active_point * 27
    Int* d_ori_index,                            // size = num_active_point
    Int* d_active_output_point_blockid,          // size = num_active_point
    Int* d_block_output_address,                 // size = num_block * max_point_per_block
    Int* d_block_active_output_count,            // size = num_block
    Int* d_block_input_address_list,             // size = num_block
    Int* d_block_input_address_list_size,        // size = num_block + 1
    
    short* d_output_rulebook_backward,                      // size = num_block * max_point_per_block * 27
    
    Int num_active_point,
    Int num_block,
    Int max_point_per_block,
    Int max_input_address,
    Int ctr,
    Int use_normal
);

void dgetSplitPointListFlag(uint32_t* d_out_flag,    
							Int* d_blockid,     
							Int* d_output_input_address_size,
							Int NActivePoints,
							Int Threshold);

void dIncreGetSplitPointListFlag(uint32_t* d_out_flag,    
                            Int* d_blockid,     
                            Int* d_block_input_address_list_size, 
                            Int* d_chunk_points_num,   
                            Int NActivePoints,
                            Int Threshold);

void dGetChunkQueryList(Int* d_block_grids,
                        Int num_block,
                        Int chunk_size,
                        Int* d_chunk_query_list,
                        Int d_chunk_query_list_size);

void dGetFlag(Int* d_chunk_result_list, 
            uint32_t* d_chunk_result_flag,
            Int* d_chunk_result_count,
            Int num_block,
            Int max_point_per_block);

template <Int Dimension>
class SubmanifoldChuckBuildObject
{
    public:
    Int NActivePoints;
    Int conv_size;
    Int volume;
    Int ctr;
    Int chunk_size;
    Int num_block;
    Int use_normal;
    /* Device Memory */
    Int *d_global_points = NULL; /*input points| NActivePoints * Dimension*/
    // Int *d_global_addresses = NULL;
    Int *d_global_query_list = NULL;     /* input regions hash key| NActivePoints * Dimension * volume */
    Int *d_global_query_result = NULL;   /*address and num| NActivePoints * volume*/
    Int *d_global_points_blockid = NULL; /* points chunk id| NActivepoints */
    Int *d_global_points_ori_index = NULL; /* NActivepoints */

    uint32_t *d_split_point_list_flag = NULL;
    uint32_t *d_split_point_list_flag_long = NULL;   //size = NActivePoints * 3
    size_t *d_split_point_list_size = NULL;
    Int *d_split_point_list = NULL; /* NActivePoints * Dimension */
    Int *d_split_point_ori_index = NULL;

    Int *d_global_points_from_split = NULL;  /* NActivePoints */

    torch::Tensor d_block_input_address_list; // concatenated chunk points indexs. 
    Int* d_block_input_address_list_size = NULL; // chunk start indexs in 'd_block_input_address_list'

    /* High-level structure*/
    GPU_SparseGrid<Dimension> &SG;
    Compacting_Point_Hashtable<Dimension> block_hashtable;



    SubmanifoldChuckBuildObject(Int NActivePoints_, long* conv_size_, Int chunk_size_, GPU_SparseGrid<Dimension> &SG_, Int use_normal_) : SG(SG_)
    {
        if (Dimension != 3)
        {
            fprintf(stderr, "dimension != 3\n");
            abort();
        }
        NActivePoints = NActivePoints_;
        conv_size = (Int)conv_size_[0];
        assert(conv_size == 3);
        volume = 1;
        for(Int i = 0; i < Dimension; i++) {
          volume *= (Int)conv_size_[i];
        }

        d_global_points = SG.pHash->getAllPoints();
        ctr = SG.ctr;

        chunk_size = chunk_size_;
        num_block = 0;
        use_normal = use_normal_;

        initialize(NActivePoints_);
    }

    SubmanifoldChuckBuildObject(SubmanifoldChuckBuildObject& obj, Int chunk_size_) : SG(obj.SG)
    {
        // 1. get pre not included NActivePoints 
        NActivePoints = obj.getSplitPointList();
        conv_size = obj.conv_size;
        assert(conv_size == 3);
        volume = obj.volume;
        ctr = SG.ctr;
        chunk_size = chunk_size_;
        num_block = 0;
        use_normal = obj.use_normal;

        // 2. copy points and ori_index
        initialize(NActivePoints);

        gpuErrchk(cudaMalloc((void **)&d_global_points_from_split, sizeof(Int) * NActivePoints * Dimension));
        gpuErrchk(cudaMemcpy(d_global_points_from_split, obj.d_split_point_list, sizeof(Int) * NActivePoints * Dimension, cudaMemcpyDeviceToDevice));
        if(use_normal)        
            gpuErrchk(cudaMemcpy(d_global_points_ori_index, obj.d_split_point_ori_index, sizeof(Int) * NActivePoints, cudaMemcpyDeviceToDevice));
        d_global_points = d_global_points_from_split;
    }

    void initialize(Int length)
    {
        gpuErrchk(cudaMalloc((void **)&d_global_query_list, sizeof(Int) * NActivePoints * volume));
        gpuErrchk(cudaMalloc((void **)&d_global_query_result, sizeof(Int) * NActivePoints * volume));
        gpuErrchk(cudaMalloc((void **)&d_global_points_blockid, sizeof(Int) * NActivePoints));
        if(use_normal)
            gpuErrchk(cudaMalloc((void **)&d_global_points_ori_index, sizeof(Int) * NActivePoints));
    }

    ~SubmanifoldChuckBuildObject()
    {
        if (d_global_query_list != NULL)
        {
            gpuErrchk(cudaFree(d_global_query_list));
        }
        if (d_global_query_result != NULL)
        {
            gpuErrchk(cudaFree(d_global_query_result));
        }
        if (d_global_points_blockid != NULL)
        {
            gpuErrchk(cudaFree(d_global_points_blockid));
        }
        if (d_global_points_ori_index != NULL)
        {
            gpuErrchk(cudaFree(d_global_points_ori_index));
        }
        if (d_split_point_list_flag != NULL)
        {
            gpuErrchk(cudaFree(d_split_point_list_flag));
        }
        if (d_split_point_list_flag_long != NULL)
        {
            gpuErrchk(cudaFree(d_split_point_list_flag_long));
        }
        if (d_split_point_list_size != NULL)
        {
            gpuErrchk(cudaFree(d_split_point_list_size));
        }
        if (d_split_point_list != NULL)
        {
            gpuErrchk(cudaFree(d_split_point_list));
        }
        if (d_split_point_ori_index != NULL)
        {
            gpuErrchk(cudaFree(d_split_point_ori_index));
        }
        if (d_global_points_from_split != NULL)
        {
            gpuErrchk(cudaFree(d_global_points_from_split));
        }
        if (d_block_input_address_list_size != NULL) 
        {
            gpuErrchk(cudaFree(d_block_input_address_list_size));
        }


    }


    void generateQueryList()
    {
    /*
        get input region of output
        @input:
            d_global_points: output points
        @output:
            d_global_query_list: input regions hash key.
    */
        EASY_FUNCTION(profiler::colors::Cyan);

        // Generate query list on GPU, save hash to d_global_query_list(NActivePoints * 27)
        dGetPointsWithinInputRegion(d_global_points, d_global_query_list, NActivePoints, conv_size, Dimension);
    }

    void global_hash_query()
    {
    /*
        from input region hash keys to query results 
        @input:
            d_global_query_list: input region hash keys
        @output:
            d_global_query_result: address of each points, and number of this points.
    */

        EASY_FUNCTION(profiler::colors::Cyan);

        assert(NActivePoints > 0);
        // global_hashtable.retrieve((uint32_t *)d_global_query_list, (uint32_t *)d_global_query_result, NActivePoints * volume);
        SG.pHash->retrieve_key((uint32_t *)d_global_query_list, (uint32_t *)d_global_query_result, NActivePoints * volume);   
    }
    


    void get_chunk_input_address()    {
        EASY_FUNCTION(profiler::colors::Cyan);
        EASY_BLOCK("Retrieve"); 

        /* For each chunk, we first query all points within 
        * possible input region, i.e., chunk region + margin.
        * Then perform cudpp compacting, in order to eliminate 
        * inactive points.
        * 
        * @input: list of chunk spatial location,
        *         chunk number,
        *         global hashtable,
        *         
        * @output: list of input address list for all chunks,
        *          length of input address list
        */
        Int max_point_per_block = (chunk_size+2) * (chunk_size+2) * (chunk_size+2);

        // gpuErrchk(cudaMalloc((void **)&d_block_input_address_list, sizeof(Int) * max_point_per_block * num_block));
        // gpuErrchk(cudaMalloc((void **)&d_block_input_address_list_size, sizeof(Int) * (num_block + 1)));

        Int* d_chunk_query_list = NULL;
        Int* d_chunk_result_list = NULL;
        Int* d_chunk_result_flag = NULL;

        gpuErrchk(cudaMalloc((void **)&d_chunk_query_list, sizeof(Int) * max_point_per_block * num_block));
        gpuErrchk(cudaMalloc((void **)&d_chunk_result_list, sizeof(Int) * max_point_per_block * num_block));
        gpuErrchk(cudaMalloc((void **)&d_chunk_result_flag, sizeof(Int) * max_point_per_block * num_block));

        // dGetChunkQueryList(Int* d_chunk_grids, Int num_block);
#ifdef PRINT_CHUNK
        printf("chunk list: %d %d %d\r\n", chunk_size,max_point_per_block,num_block);
        printf("size = %d\n", block_hashtable.size);
#endif
        // get chunk input location hash(each input region continuously), save to d_chunk_query_list(chunk num * (chunksize+2)^3)
        dGetChunkQueryList(block_hashtable.getAllPoints(),
                           block_hashtable.size,
                           chunk_size,
                           d_chunk_query_list,
                           max_point_per_block * num_block);

        #if 0
        Int* h_chunk_query_list = new Int[max_point_per_block * num_block * Dimension];
        gpuErrchk(cudaMemcpy(h_chunk_query_list, d_chunk_query_list, sizeof(Int) * max_point_per_block * num_block * Dimension, cudaMemcpyDeviceToHost));

        for (Int i = 0; i < max_point_per_block * num_block; i++)  {
            printf("%d %d %d\n", h_chunk_query_list[i], h_chunk_query_list[i + max_point_per_block * num_block], h_chunk_query_list[i + 2 * max_point_per_block * num_block]);
        }
        printf("\n");
        delete [] h_chunk_query_list;

        #endif
        // get input points location in output location.
        SG.pHash->retrieve_key((uint32_t* )d_chunk_query_list,
                           (uint32_t* )d_chunk_result_list, 
                           max_point_per_block * num_block);

        auto d_chunk_result_count = torch::zeros({num_block + 1},
                                     torch::dtype(torch::kInt32).device(torch::kCUDA));
        // set invalid input points to 0, and valid to 1, save in d_chunk_result_flag, and each chunk point num in d_chunk_result_count.
        dGetFlag(d_chunk_result_list,
                 (uint32_t* )d_chunk_result_flag,
                 d_chunk_result_count.data<Int>(),
                 num_block,
                 max_point_per_block);

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;
        EASY_BLOCK("scan"); 

#if 1
        gpuErrchk(cudaMalloc((void**)&d_block_input_address_list_size, sizeof(Int) * (num_block + 1)));
                                           
        Int *h_chunk_result_count = new Int[num_block + 1];
        Int *h_block_input_address_list_size = new Int[num_block + 1];
        gpuErrchk(cudaMemcpy(h_chunk_result_count, d_chunk_result_count.data<Int>(), sizeof(Int) * (num_block + 1), cudaMemcpyDeviceToHost));
        h_block_input_address_list_size[0] = 0;
        for (int i = 1; i < num_block + 1; i++)
            h_block_input_address_list_size[i] = h_block_input_address_list_size[i - 1] + h_chunk_result_count[i - 1];
        // Scan ADD
        // get each chunk start location by prefix sum of d_chunk_result_count.
        // each chunk include its outer ring
        gpuErrchk(cudaMemcpy(d_block_input_address_list_size, h_block_input_address_list_size, sizeof(Int) * (num_block + 1), cudaMemcpyHostToDevice));
        delete[] h_chunk_result_count;
        delete[] h_block_input_address_list_size;
        
#endif

        EASY_END_BLOCK;
        EASY_BLOCK("compacting"); 


#ifdef PRINT_CHUNK
        std::cout << d_block_input_address_list_size_tensor << std::endl;
#endif
        // get real exist input location
        d_block_input_address_list = torch::empty({max_point_per_block * num_block}, torch::dtype(torch::kInt32).device(torch::kCUDA));

        auto d_num = torch::empty({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));

        CUDPP_Compacting compacting(max_point_per_block * num_block);

        compacting.apply(d_block_input_address_list.data<Int>(), 
                        (size_t*) d_num.data<int64_t>(),
                        (uint32_t *)d_chunk_result_list,
                        (uint32_t *)d_chunk_result_flag, 
                        max_point_per_block * num_block);
        d_num = d_num.cpu();
        // reduce input location
        d_block_input_address_list.resize_({d_num[0].item<int64_t>()});
        // reduce input location

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;
        // DEBUG
        #if 0
        Int* h_block_input_address_list = new Int[max_point_per_block * num_block];
        Int* h_block_input_address_list_size = new Int[num_block];
        gpuErrchk(cudaMemcpy(h_block_input_address_list, d_block_input_address_list, sizeof(Int) * max_point_per_block * num_block, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_block_input_address_list_size, d_block_input_address_list_size, sizeof(Int) * (num_block + 1), cudaMemcpyDeviceToHost));

        for(Int i = 0; i < num_block; i++)   {
            printf("Block : %d\n", i);
            for(Int j = h_block_input_address_list_size[i]; j < h_block_input_address_list_size[i+1]; j++)  {
                printf("%d ", h_block_input_address_list[j]);
            }
            printf("\n");
        }

        delete [] h_block_input_address_list;
        delete [] h_block_input_address_list_size;
        #endif
        
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        gpuErrchk(cudaFree(d_chunk_query_list));
        gpuErrchk(cudaFree(d_chunk_result_list));
        gpuErrchk(cudaFree(d_chunk_result_flag));
    }

    void get_block_id()
    {
        EASY_FUNCTION(profiler::colors::Cyan);

        /*
        * 1. point -> 16x block grid
        * 2. compacting hashing
        * 3. retrieve
        * 4. Split:
        *       
        * */
        Int *d_global_block_grids = NULL;
        gpuErrchk(cudaMalloc((void **)&d_global_block_grids, sizeof(Int) * NActivePoints * Dimension));
        // global points coords to block grids coords.
        dGetBlockGrid(d_global_points, d_global_block_grids, chunk_size, NActivePoints * Dimension);
        block_hashtable.insert((uint32_t *)d_global_block_grids, NActivePoints);
        // get unique id of every block coords.
        block_hashtable.retrieve((uint32_t *)d_global_block_grids, (uint32_t *)d_global_points_blockid, NActivePoints);

        num_block = block_hashtable.getCompactingSize();
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        gpuErrchk(cudaFree(d_global_block_grids));
    }

    void get_ori_index(const std::vector<Float3> &normal)    {
        EASY_FUNCTION(profiler::colors::Cyan);

        assert(use_normal);

        std::vector<Int> oriIndex(NActivePoints);
        // TODO: can use openmp to parallize
        for(Int i = 0; i < NActivePoints; i++)
        {
            const Float3 &n = normal[ctr + i];
            oriIndex[i] = OrientedFilter(n);
        }
        gpuErrchk(cudaMemcpy(d_global_points_ori_index, oriIndex.data(), sizeof(Int) * NActivePoints, cudaMemcpyHostToDevice));
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
    }

Int generate_rulebook(RBChunkPointerList& new_rules)
    {
        EASY_FUNCTION(profiler::colors::Cyan);
        Int max_point_per_block = (chunk_size+2) * (chunk_size+2) * (chunk_size+2);
        // int my_cuda_mem_new_count;
        // my_cuda_mem_new_count++;
        Int *d_block_active_output_count = NULL;
        short *d_output_rulebook = NULL;
        short *d_output_rulebook_backward = NULL;
        Int *d_output_input_address = NULL;
        Int *d_output_output_address = NULL;
        gpuErrchk(cudaMalloc((void **)&d_block_active_output_count, sizeof(Int) * num_block));
        gpuErrchk(cudaMemset(d_block_active_output_count, 0, sizeof(Int) * num_block));

        // this MAX_INPUT_ADDRESS could cause problem if set too small.
        gpuErrchk(cudaMalloc((void **)&d_output_rulebook, sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27));
        gpuErrchk(cudaMemset(d_output_rulebook, 0xFFFF, sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27));
#if 0
        gpuErrchk(cudaMalloc((void **)&d_output_rulebook_backward, sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27));
        gpuErrchk(cudaMemset(d_output_rulebook_backward, 0xFFFF, sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27));
#endif
        gpuErrchk(cudaMalloc((void **)&d_output_input_address, sizeof(Int) * num_block * max_point_per_block));

        gpuErrchk(cudaMalloc((void **)&d_output_output_address, sizeof(Int) * num_block * max_point_per_block));

        EASY_BLOCK("dGenerateChuckRulebook"); // Begin block with default color == Amber100
        dGenerateChuckRulebook(d_global_query_result,
                                d_global_points_ori_index,
                                d_global_points_blockid,
                                d_block_active_output_count,
                                d_block_input_address_list.data<Int>(),
                                d_block_input_address_list_size,
                                d_output_rulebook,
                                d_output_output_address,
                                d_output_input_address,
                                NActivePoints,
                                num_block,
                                max_point_per_block,
                                MAX_INPUT_ADDRESS,
                                ctr,
                                use_normal);
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;
#if 0
    dGenerateChunkBackwardRB(
        d_global_query_result,             // size = num_active_point * 27
        d_global_points_ori_index,                            // size = num_active_point
        d_global_points_blockid,          // size = num_active_point
        d_output_output_address,                 // size = num_block * max_point_per_block
        d_block_active_output_count,            // size = num_block
        d_block_input_address_list,             // size = num_block
        d_block_input_address_list_size,        // size = num_block + 1
        
        d_output_rulebook_backward,                      // size = num_block * max_point_per_block * 27
        
        NActivePoints,
        num_block,
        max_point_per_block,
        MAX_INPUT_ADDRESS,
        ctr,
        use_normal
    );
#endif
#if 1
        EASY_BLOCK("Copy to RBChuckTensor"); // Begin block with default color == Amber100

        // Copy to RBChuckTensor
        Int countActiveOutput = 0;

        Int *h_output_input_address_size = new Int[num_block + 1];
        Int *h_block_active_output_count = new Int[num_block];

        gpuErrchk(cudaMemcpy(h_output_input_address_size, d_block_input_address_list_size, sizeof(Int) * (num_block + 1), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_block_active_output_count, d_block_active_output_count, sizeof(Int) * num_block, cudaMemcpyDeviceToHost));
        
        // if (chunk_size == 8) {
        //     printf("now input size: %d\n", h_output_input_address_size[num_block]);
        //     EASY_VALUE("now input size", h_output_input_address_size[num_block], EASY_VIN("now"));
        // }


        for (Int i = 0; i < num_block; i++) {
            Int inputAddress_size = h_output_input_address_size[i + 1] - h_output_input_address_size[i];
            Int outputAddress_size = h_block_active_output_count[i];
            Int rules_size = outputAddress_size * 27;
            Int backward_rules_size = inputAddress_size * 27;
#ifdef PRINT_CHUNK
            printf("\tblock = %d\n", i);
            printf("\tnum_block = %d\n", num_block);
            printf("\tinputAddress_size = %d\n", inputAddress_size);
            printf("\toutputAddress_size = %d\n", outputAddress_size);
            printf("\trules_size = %d\n", rules_size);
#endif
            if (inputAddress_size > MAX_INPUT_ADDRESS)
                continue;
            countActiveOutput += outputAddress_size;
            //
            InputAddress rbp = InputAddress(
                (void*)&(d_output_input_address[h_output_input_address_size[i]]),
                (void*)&(d_output_output_address[i * max_point_per_block]),
                (void*)&(d_output_rulebook[MAX_INPUT_ADDRESS * 27 * i]),
                NULL,
                inputAddress_size,
                outputAddress_size);
            new_rules.list.push_back(rbp);
        }
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk(cudaDeviceSynchronize());
#endif
        EASY_END_BLOCK;

        delete[] h_output_input_address_size;
        delete[] h_block_active_output_count;

#endif

        new_rules.cuda_mem_to_release.push_back(d_block_active_output_count);
        new_rules.cuda_mem_to_release_short.push_back(d_output_rulebook);
        //new_rules.cuda_mem_to_release_short.push_back(d_output_rulebook_backward);
        new_rules.cuda_mem_to_release.push_back(d_output_input_address);
        new_rules.cuda_mem_to_release.push_back(d_output_output_address);

#ifdef PRINT_MEM_ALLOC
        new_rules.size += sizeof(Int) * num_block;   //d_block_active_output_count
        new_rules.size += sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27; //d_output_rulebook
        new_rules.size += sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27; //d_output_rulebook_backward
        new_rules.size += sizeof(Int) * num_block * max_point_per_block;  //d_output_input_address
        new_rules.size += sizeof(Int) * num_block * max_point_per_block;  //d_output_output_address
#endif

        return countActiveOutput;
        // return 0;
    }

    Int getSplitPointList()
    {

        EASY_FUNCTION(profiler::colors::Cyan);
        size_t split_point_list_size;

        gpuErrchk(cudaMalloc((void **)&d_split_point_list_flag, sizeof(uint32_t) * NActivePoints));
        gpuErrchk(cudaMalloc((void **)&d_split_point_list_flag_long, sizeof(uint32_t) * NActivePoints * Dimension));
        gpuErrchk(cudaMalloc((void **)&d_split_point_list_size, sizeof(size_t)));
        gpuErrchk(cudaMalloc((void **)&d_split_point_list, sizeof(Int) * NActivePoints * Dimension));
        if(use_normal)
            gpuErrchk(cudaMalloc((void **)&d_split_point_ori_index, sizeof(Int) * NActivePoints));

        dgetSplitPointListFlag(d_split_point_list_flag,    
                            d_global_points_blockid,     
                            d_block_input_address_list_size,
                            NActivePoints,
                            MAX_INPUT_ADDRESS);

        gpuErrchk(cudaMemcpy(d_split_point_list_flag_long, d_split_point_list_flag, sizeof(uint32_t) * NActivePoints, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_split_point_list_flag_long + NActivePoints, d_split_point_list_flag, sizeof(uint32_t) * NActivePoints, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_split_point_list_flag_long + NActivePoints * 2, d_split_point_list_flag, sizeof(uint32_t) * NActivePoints, cudaMemcpyDeviceToDevice));
        
        CUDPP_Compacting compacting(NActivePoints * Dimension);
        // get not included points
        compacting.apply(d_split_point_list, d_split_point_list_size, d_global_points, d_split_point_list_flag_long, NActivePoints * Dimension);
        
        gpuErrchk(cudaMemcpy(&split_point_list_size, d_split_point_list_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        // not included points size
        Int return_val = (Int)split_point_list_size / 3;
        
        if(use_normal)
            compacting.apply(d_split_point_ori_index, d_split_point_list_size, d_global_points_ori_index, d_split_point_list_flag, NActivePoints);
#ifdef PRINT_CHUNK
        printf("split_point_list_size = %d\n", return_val);
#endif
        return return_val;
    }
};

/****************************************************************************************************************************************/
template <Int Dimension>
class IncreSubmanifoldChuckBuildObject
{
    public:
    size_t NActivePoints;
    Int conv_size;
    Int volume;
    Int ctr;
    Int chunk_size;
    Int num_block;
    Int use_normal;
    /* Device Memory */
    Int *d_global_points = NULL; /*input points| NActivePoints * Dimension*/
    // Int *d_global_addresses = NULL;
    Int *d_global_query_list = NULL;     /* input regions hash key| NActivePoints * Dimension * volume */
    Int *d_global_query_result = NULL;   /*address and num| NActivePoints * volume*/
    Int *d_global_points_blockid = NULL; /* points chunk id| NActivepoints */
    Int *d_global_points_ori_index = NULL; /* NActivepoints */

    uint32_t *d_split_point_list_flag = NULL;
    uint32_t *d_split_point_list_flag_long = NULL;   //size = NActivePoints * 3
    size_t *d_split_point_list_size = NULL;
    Int *d_split_point_list = NULL; /* NActivePoints * Dimension */
    Int *d_split_point_ori_index = NULL;

    Int *d_global_points_from_split = NULL;  /* NActivePoints */
    Int *d_chunk_points_num = NULL;

    torch::Tensor d_block_input_address_list; // concatenated chunk points indexs. 
    Int * d_block_input_address_list_size = NULL; // chunk start indexs in 'd_block_input_address_list'

    /* High-level structure*/
    GPU_SparseGrid<Dimension> &SG;
    GPU_SparseGrid<Dimension> &pre_SG;
    Compacting_Point_Hashtable<Dimension> block_hashtable;

    Int *d_reduced_points = NULL; 
    size_t *d_reduced_num = NULL;
    Int *d_output_query_result = NULL;

    IncreSubmanifoldChuckBuildObject(size_t NActivePoints_, long* conv_size_, Int chunk_size_, GPU_SparseGrid<Dimension> &SG_, Int use_normal_, 
    GPU_SparseGrid<Dimension> &pre_SG_) : SG(SG_), pre_SG(pre_SG_)
    {
        if (Dimension != 3)
        {
            fprintf(stderr, "dimension != 3\n");
            abort();
        }
        NActivePoints = NActivePoints_;
        conv_size = (Int)conv_size_[0];
        assert(conv_size == 3);
        volume = 1;
        for(Int i = 0; i < Dimension; i++) {
          volume *= (Int)conv_size_[i];
        }

        d_global_points = SG.pHash->getAllPoints();

        // delete pre exist points        
        Int *d_pre_not_exist = NULL;
        gpuErrchk(cudaMalloc((void**)&d_pre_not_exist, sizeof(Int)*NActivePoints));
        pre_SG.pHash->retrieve((uint32_t*) d_global_points, (uint32_t*) d_pre_not_exist, NActivePoints);
        dGetNot(d_pre_not_exist, NActivePoints);
        uint32_t *d_pre_not_exist_long = NULL;
        gpuErrchk(cudaMalloc((void**)&d_pre_not_exist_long, sizeof(uint32_t)*NActivePoints * Dimension));
        gpuErrchk(cudaMemcpy(d_pre_not_exist_long, d_pre_not_exist, sizeof(uint32_t)*NActivePoints,cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_pre_not_exist_long + NActivePoints, d_pre_not_exist, sizeof(uint32_t)*NActivePoints, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_pre_not_exist_long + 2 * NActivePoints, d_pre_not_exist, sizeof(uint32_t)*NActivePoints, cudaMemcpyDeviceToDevice));
        CUDPP_Compacting compacting(NActivePoints * Dimension);
        gpuErrchk(cudaMalloc((void**)&d_reduced_points, sizeof(Int)*NActivePoints*Dimension));
        gpuErrchk(cudaMalloc((void**)&d_reduced_num, sizeof(size_t)));
        compacting.apply(d_reduced_points, d_reduced_num, d_global_points, d_pre_not_exist_long, NActivePoints*Dimension);
        
        
        d_global_points = d_reduced_points;
        gpuErrchk(cudaMemcpy(&NActivePoints, d_reduced_num, sizeof(size_t), cudaMemcpyDeviceToHost));
        NActivePoints /= 3;
        
        
        gpuErrchk(cudaFree(d_pre_not_exist));
        gpuErrchk(cudaFree(d_pre_not_exist_long));

        ctr = SG.ctr;

        chunk_size = chunk_size_;
        num_block = 0;
        use_normal = use_normal_;

        initialize(NActivePoints);
    }

    IncreSubmanifoldChuckBuildObject(IncreSubmanifoldChuckBuildObject& obj, Int chunk_size_) : SG(obj.SG), pre_SG(obj.pre_SG)
    {
        // 1. get NActivePoints
        NActivePoints = obj.getSplitPointList();
        conv_size = obj.conv_size;
        assert(conv_size == 3);
        volume = obj.volume;
        ctr = SG.ctr;
        chunk_size = chunk_size_;
        num_block = 0;
        use_normal = obj.use_normal;

        // 2. copy points and ori_index
        initialize(NActivePoints);

        gpuErrchk(cudaMalloc((void **)&d_global_points_from_split, sizeof(Int) * NActivePoints * Dimension));
        gpuErrchk(cudaMemcpy(d_global_points_from_split, obj.d_split_point_list, sizeof(Int) * NActivePoints * Dimension, cudaMemcpyDeviceToDevice));
        if(use_normal)
            gpuErrchk(cudaMemcpy(d_global_points_ori_index, obj.d_split_point_ori_index, sizeof(Int) * NActivePoints, cudaMemcpyDeviceToDevice));
        d_global_points = d_global_points_from_split;
    }

    void initialize(Int length)
    {
        gpuErrchk(cudaMalloc((void **)&d_global_query_list, sizeof(Int) * NActivePoints * volume));
        gpuErrchk(cudaMalloc((void **)&d_global_query_result, sizeof(Int) * NActivePoints * volume));
        gpuErrchk(cudaMalloc((void **)&d_global_points_blockid, sizeof(Int) * NActivePoints));
        gpuErrchk(cudaMalloc((void **)&d_output_query_result, sizeof(Int) * NActivePoints));
        if(use_normal)
            gpuErrchk(cudaMalloc((void **)&d_global_points_ori_index, sizeof(Int) * NActivePoints));
    }

    ~IncreSubmanifoldChuckBuildObject()
    {
        if (d_global_query_list != NULL)
        {
            gpuErrchk(cudaFree(d_global_query_list));
        }
        if (d_global_query_result != NULL)
        {
            gpuErrchk(cudaFree(d_global_query_result));
        }
        if (d_global_points_blockid != NULL)
        {
            gpuErrchk(cudaFree(d_global_points_blockid));
        }
        if (d_global_points_ori_index != NULL)
        {
            gpuErrchk(cudaFree(d_global_points_ori_index));
        }
        if (d_split_point_list_flag != NULL)
        {
            gpuErrchk(cudaFree(d_split_point_list_flag));
        }
        if (d_split_point_list_flag_long != NULL)
        {
            gpuErrchk(cudaFree(d_split_point_list_flag_long));
        }
        if (d_split_point_list_size != NULL)
        {
            gpuErrchk(cudaFree(d_split_point_list_size));
        }
        if (d_split_point_list != NULL)
        {
            gpuErrchk(cudaFree(d_split_point_list));
        }
        if (d_split_point_ori_index != NULL)
        {
            gpuErrchk(cudaFree(d_split_point_ori_index));
        }
        if (d_global_points_from_split != NULL)
        {
            gpuErrchk(cudaFree(d_global_points_from_split));
        }
        if (d_reduced_points != NULL)
        {
            gpuErrchk(cudaFree(d_reduced_points));
        }
        if (d_reduced_num != NULL)
        {
            gpuErrchk(cudaFree(d_reduced_num));
        }
        if (d_output_query_result != NULL)
        {
            gpuErrchk(cudaFree(d_output_query_result));
        }
        if (d_block_input_address_list_size != NULL) 
        {
            gpuErrchk(cudaFree(d_block_input_address_list_size));
        }

        if (d_chunk_points_num != NULL)
        {
            gpuErrchk(cudaFree(d_chunk_points_num));
        }

    }


    void generateQueryList()
    {
    /*
        get input region of output
        @input:
            d_global_points: output points
        @output:
            d_global_query_list: input regions hash key.
    */
        if (NActivePoints == 0) {
            return;
        }
        EASY_FUNCTION(profiler::colors::Cyan);

        // Generate query list on GPU, save hash to d_global_query_list(NActivePoints * 27)
        dGetPointsWithinInputRegion(d_global_points, d_global_query_list, NActivePoints, conv_size, Dimension);
    }

    void global_hash_query()
    {
    /*
        from input region hash keys to query results 
        @input:
            d_global_query_list: input region hash keys
        @output:
            d_global_query_result: address of each points.
    */

        EASY_FUNCTION(profiler::colors::Cyan);
        if (NActivePoints == 0) {
            return ;
        }
        // global_hashtable.retrieve((uint32_t *)d_global_query_list, (uint32_t *)d_global_query_result, NActivePoints * volume);
        pre_SG.pHash->retrieve_key((uint32_t *)d_global_query_list, (uint32_t *)d_global_query_result, NActivePoints * volume);   

        SG.pHash->retrieve((uint32_t *)d_global_points, (uint32_t*) d_output_query_result, NActivePoints);

    }
    


    void get_chunk_input_address()    {
        if (NActivePoints == 0) {
            return;
        }
        EASY_FUNCTION(profiler::colors::Cyan);
        EASY_BLOCK("Retrieve"); 

        /* For each chunk, we first query all points within 
        * possible input region, i.e., chunk region + margin.
        * Then perform cudpp compacting, in order to eliminate 
        * inactive points.
        * 
        * @input: list of chunk spatial location,
        *         chunk number,
        *         global hashtable,
        *         
        * @output: list of input address list for all chunks,
        *          length of input address list
        */
        Int max_point_per_block = (chunk_size+2) * (chunk_size+2) * (chunk_size+2);

        // gpuErrchk(cudaMalloc((void **)&d_block_input_address_list, sizeof(Int) * max_point_per_block * num_block));
        // gpuErrchk(cudaMalloc((void **)&d_block_input_address_list_size, sizeof(Int) * (num_block + 1)));

        Int* d_chunk_query_list = NULL;
        Int* d_chunk_result_list = NULL;
        Int* d_chunk_result_flag = NULL;

        gpuErrchk(cudaMalloc((void **)&d_chunk_query_list, sizeof(Int) * max_point_per_block * num_block));
        gpuErrchk(cudaMalloc((void **)&d_chunk_result_list, sizeof(Int) * max_point_per_block * num_block));
        gpuErrchk(cudaMalloc((void **)&d_chunk_result_flag, sizeof(Int) * max_point_per_block * num_block));

        // dGetChunkQueryList(Int* d_chunk_grids, Int num_block);
#ifdef PRINT_CHUNK
        printf("chunk list: %d %d %d\r\n", chunk_size,max_point_per_block,num_block);
        printf("size = %d\n", block_hashtable.size);
#endif
        // get chunk input location hash(each input region continuously), save to d_chunk_query_list(chunk num * (chunksize+2)^3)
        dGetChunkQueryList(block_hashtable.getAllPoints(),
                           block_hashtable.size,
                           chunk_size,
                           d_chunk_query_list,
                           max_point_per_block * num_block);

        #if 0
        Int* h_chunk_query_list = new Int[max_point_per_block * num_block * Dimension];
        gpuErrchk(cudaMemcpy(h_chunk_query_list, d_chunk_query_list, sizeof(Int) * max_point_per_block * num_block * Dimension, cudaMemcpyDeviceToHost));

        for (Int i = 0; i < max_point_per_block * num_block; i++)  {
            printf("%d %d %d\n", h_chunk_query_list[i], h_chunk_query_list[i + max_point_per_block * num_block], h_chunk_query_list[i + 2 * max_point_per_block * num_block]);
        }
        printf("\n");
        delete [] h_chunk_query_list;

        #endif
        // get input points location in output location.
        pre_SG.pHash->retrieve_key((uint32_t* )d_chunk_query_list,
                           (uint32_t* )d_chunk_result_list, 
                           max_point_per_block * num_block);

        auto d_chunk_result_count = torch::zeros({num_block + 1},
                                     torch::dtype(torch::kInt32).device(torch::kCUDA));
        // set invalid input points to 0, and valid to 1, save in d_chunk_result_flag, and each chunk point num in d_chunk_result_count.
        dGetFlag(d_chunk_result_list,
                 (uint32_t* )d_chunk_result_flag,
                 d_chunk_result_count.data<Int>(),
                 num_block,
                 max_point_per_block);

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;
        EASY_BLOCK("scan"); 

#if 1
        gpuErrchk(cudaMalloc((void**)&d_block_input_address_list_size, sizeof(Int) * (num_block + 1)));
                                           
        Int *h_chunk_result_count = new Int[num_block + 1];
        Int *h_block_input_address_list_size = new Int[num_block + 1];
        gpuErrchk(cudaMemcpy(h_chunk_result_count, d_chunk_result_count.data<Int>(), sizeof(Int) * (num_block + 1), cudaMemcpyDeviceToHost));
        h_block_input_address_list_size[0] = 0;
        for (int i = 1; i < num_block + 1; i++)
            h_block_input_address_list_size[i] = h_block_input_address_list_size[i - 1] + h_chunk_result_count[i - 1];
        // Scan ADD
        // get each chunk start location by prefix sum of d_chunk_result_count.
        // each chunk include its outer ring
        gpuErrchk(cudaMemcpy(d_block_input_address_list_size, h_block_input_address_list_size, sizeof(Int) * (num_block + 1), cudaMemcpyHostToDevice));
        delete[] h_chunk_result_count;
        delete[] h_block_input_address_list_size;
        
        
#endif
        EASY_END_BLOCK;
        EASY_BLOCK("compacting"); 


#ifdef PRINT_CHUNK
        std::cout << d_block_input_address_list_size_tensor << std::endl;
#endif
        // get real exist input location
        d_block_input_address_list = torch::empty({max_point_per_block * num_block}, torch::dtype(torch::kInt32).device(torch::kCUDA));

        auto d_num = torch::empty({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));

        CUDPP_Compacting compacting(max_point_per_block * num_block);

        compacting.apply(d_block_input_address_list.data<Int>(), 
                        (size_t*) d_num.data<int64_t>(),
                        (uint32_t *)d_chunk_result_list,
                        (uint32_t *)d_chunk_result_flag, 
                        max_point_per_block * num_block);
        
        d_num = d_num.cpu();
        // reduce input location
        d_block_input_address_list.resize_({d_num[0].item<int64_t>()});

#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;
        // DEBUG
        #if 0
        Int* h_block_input_address_list = new Int[max_point_per_block * num_block];
        Int* h_block_input_address_list_size = new Int[num_block];
        gpuErrchk(cudaMemcpy(h_block_input_address_list, d_block_input_address_list, sizeof(Int) * max_point_per_block * num_block, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_block_input_address_list_size, d_block_input_address_list_size, sizeof(Int) * (num_block + 1), cudaMemcpyDeviceToHost));

        for(Int i = 0; i < num_block; i++)   {
            printf("Block : %d\n", i);
            for(Int j = h_block_input_address_list_size[i]; j < h_block_input_address_list_size[i+1]; j++)  {
                printf("%d ", h_block_input_address_list[j]);
            }
            printf("\n");
        }

        delete [] h_block_input_address_list;
        delete [] h_block_input_address_list_size;
        #endif
        
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        gpuErrchk(cudaFree(d_chunk_query_list));
        gpuErrchk(cudaFree(d_chunk_result_list));
        gpuErrchk(cudaFree(d_chunk_result_flag));
    }

    void get_block_id()
    {
        if (NActivePoints == 0) {
            return;
        }
        EASY_FUNCTION(profiler::colors::Cyan);

        /*
        * 1. point -> 16x block grid
        * 2. compacting hashing
        * 3. retrieve
        * 4. Split:
        *       
        * */
        Int *d_global_block_grids = NULL;
        gpuErrchk(cudaMalloc((void **)&d_global_block_grids, sizeof(Int) * NActivePoints * Dimension));
        // global points coords to block grids coords.
        dGetBlockGrid(d_global_points, d_global_block_grids, chunk_size, NActivePoints * Dimension);
        block_hashtable.insert((uint32_t *)d_global_block_grids, NActivePoints);
        // get unique id of every block coords.
        block_hashtable.retrieve((uint32_t *)d_global_block_grids, (uint32_t *)d_global_points_blockid, NActivePoints);

        num_block = block_hashtable.getCompactingSize();
        EASY_BLOCK("get chunk points num")
        gpuErrchk(cudaMalloc((void**)&d_chunk_points_num, sizeof(Int) * num_block));
        gpuErrchk(cudaMemset(d_chunk_points_num, 0, sizeof(Int) * num_block));
        dGetChunkPointsNum(d_global_points_blockid, d_chunk_points_num, NActivePoints);
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        gpuErrchk(cudaFree(d_global_block_grids));
    }

    void get_ori_index(const std::vector<Float3> &normal)    {
        EASY_FUNCTION(profiler::colors::Cyan);

        assert(use_normal);

        std::vector<Int> oriIndex(NActivePoints);
        // TODO: can use openmp to parallize
        for(Int i = 0; i < NActivePoints; i++)
        {
            const Float3 &n = normal[ctr + i];
            oriIndex[i] = OrientedFilter(n);
        }
        gpuErrchk(cudaMemcpy(d_global_points_ori_index, oriIndex.data(), sizeof(Int) * NActivePoints, cudaMemcpyHostToDevice));
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
    }

Int generate_rulebook(RBChunkPointerList& new_rules)
    {
        if (NActivePoints == 0) {
            return 0;
        }
        EASY_FUNCTION(profiler::colors::Cyan);
        Int max_point_per_block = (chunk_size+2) * (chunk_size+2) * (chunk_size+2);
        // int my_cuda_mem_new_count;
        // my_cuda_mem_new_count++;
        Int *d_block_active_output_count = NULL;
        short *d_output_rulebook = NULL;
        short *d_output_rulebook_backward = NULL;
        Int *d_output_input_address = NULL;
        Int *d_output_output_address = NULL;
        gpuErrchk(cudaMalloc((void **)&d_block_active_output_count, sizeof(Int) * num_block));
        gpuErrchk(cudaMemset(d_block_active_output_count, 0, sizeof(Int) * num_block));

        // this MAX_INPUT_ADDRESS could cause problem if set too small.
        gpuErrchk(cudaMalloc((void **)&d_output_rulebook, sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27));
        gpuErrchk(cudaMemset(d_output_rulebook, 0xFFFF, sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27));
        // gpuErrchk(cudaMalloc((void **)&d_output_rulebook_backward, sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27));
        // gpuErrchk(cudaMemset(d_output_rulebook_backward, 0xFFFF, sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27));

        gpuErrchk(cudaMalloc((void **)&d_output_input_address, sizeof(Int) * num_block * max_point_per_block));

        gpuErrchk(cudaMalloc((void **)&d_output_output_address, sizeof(Int) * num_block * max_point_per_block));

        EASY_BLOCK("dGenerateChuckRulebook"); // Begin block with default color == Amber100
        dGenerateIncreChuckRulebook(d_global_query_result,
                                d_global_points_ori_index,
                                d_output_query_result,
                                d_global_points_blockid,
                                d_block_active_output_count,
                                d_block_input_address_list.data<Int>(),
                                d_block_input_address_list_size,
                                d_chunk_points_num,
                                d_output_rulebook, // index=(block, outpoint, input region) val= input point offset in chunk in d_block_input_address_list
                                d_output_output_address, // index=(block, outpoint) val= output address 
                                d_output_input_address, // size = d_block_input_address_list.size  val= input address
                                NActivePoints,
                                num_block,
                                max_point_per_block,
                                MAX_INPUT_ADDRESS,
                                ctr,
                                use_normal);
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk( cudaDeviceSynchronize() );
#endif
        EASY_END_BLOCK;
#if 0
    dGenerateChunkBackwardRB(
        d_global_query_result,             // size = num_active_point * 27
        d_global_points_ori_index,                            // size = num_active_point
        d_global_points_blockid,          // size = num_active_point
        d_output_output_address,                 // size = num_block * max_point_per_block
        d_block_active_output_count,            // size = num_block
        d_block_input_address_list,             // size = num_block
        d_block_input_address_list_size,        // size = num_block + 1
        
        d_output_rulebook_backward,                      // size = num_block * max_point_per_block * 27
        
        NActivePoints,
        num_block,
        max_point_per_block,
        MAX_INPUT_ADDRESS,
        ctr,
        use_normal
    );
#endif
#if 1
        EASY_BLOCK("Copy to RBChuckTensor"); // Begin block with default color == Amber100

        // Copy to RBChuckTensor
        Int countActiveOutput = 0;

        Int *h_output_input_address_size = new Int[num_block + 1];
        Int *h_block_active_output_count = new Int[num_block];
        Int *h_chunk_points_num = new Int[num_block];
        gpuErrchk(cudaMemcpy(h_output_input_address_size, d_block_input_address_list_size, sizeof(Int) * (num_block + 1), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_block_active_output_count, d_block_active_output_count, sizeof(Int) * num_block, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_chunk_points_num, d_chunk_points_num, sizeof(Int) * num_block, cudaMemcpyDeviceToHost));

        // if (chunk_size == 8) {
        //     printf("pre input size: %d\n", h_output_input_address_size[num_block]);
        //     EASY_VALUE("pre input size", h_output_input_address_size[num_block], EASY_VIN("pre"));
        // }

        // every chunk has a rulebook
        for (Int i = 0; i < num_block; i++) {
            Int inputAddress_size = h_output_input_address_size[i + 1] - h_output_input_address_size[i];
            Int outputAddress_size = h_block_active_output_count[i];
            Int rules_size = outputAddress_size * 27;
            Int backward_rules_size = inputAddress_size * 27;
#ifdef PRINT_CHUNK
            printf("\tblock = %d\n", i);
            printf("\tnum_block = %d\n", num_block);
            printf("\tinputAddress_size = %d\n", inputAddress_size);
            printf("\toutputAddress_size = %d\n", outputAddress_size);
            printf("\trules_size = %d\n", rules_size);
#endif
            if (inputAddress_size > MAX_INPUT_ADDRESS or h_chunk_points_num[i] > MAX_INPUT_ADDRESS or outputAddress_size == 0 or inputAddress_size == 0 or rules_size == 0)
                continue;
            countActiveOutput += outputAddress_size;
            //
            InputAddress rbp = InputAddress(
                (void*)&(d_output_input_address[h_output_input_address_size[i]]),
                (void*)&(d_output_output_address[i * max_point_per_block]),
                (void*)&(d_output_rulebook[MAX_INPUT_ADDRESS * 27 * i]),
                NULL,
                inputAddress_size,
                outputAddress_size);
            new_rules.list.push_back(rbp);
        }
#ifdef BUILD_WITH_EASY_PROFILER
        gpuErrchk(cudaDeviceSynchronize());
#endif
        EASY_END_BLOCK;

        delete[] h_output_input_address_size;
        delete[] h_block_active_output_count;
        delete[] h_chunk_points_num;

#endif

        new_rules.cuda_mem_to_release.push_back(d_block_active_output_count);
        new_rules.cuda_mem_to_release_short.push_back(d_output_rulebook);
        // new_rules.cuda_mem_to_release_short.push_back(d_output_rulebook_backward);
        new_rules.cuda_mem_to_release.push_back(d_output_input_address);
        new_rules.cuda_mem_to_release.push_back(d_output_output_address);

#ifdef PRINT_MEM_ALLOC
        new_rules.size += sizeof(Int) * num_block;   //d_block_active_output_count
        new_rules.size += sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27; //d_output_rulebook
        new_rules.size += sizeof(short) * num_block * MAX_INPUT_ADDRESS * 27; //d_output_rulebook_backward
        new_rules.size += sizeof(Int) * num_block * max_point_per_block;  //d_output_input_address
        new_rules.size += sizeof(Int) * num_block * max_point_per_block;  //d_output_output_address
#endif

        return countActiveOutput;
        // return 0;
    }

    Int getSplitPointList()
    {

        EASY_FUNCTION(profiler::colors::Cyan);
        size_t split_point_list_size;

        gpuErrchk(cudaMalloc((void **)&d_split_point_list_flag, sizeof(uint32_t) * NActivePoints));
        gpuErrchk(cudaMalloc((void **)&d_split_point_list_flag_long, sizeof(uint32_t) * NActivePoints * Dimension));
        gpuErrchk(cudaMalloc((void **)&d_split_point_list_size, sizeof(size_t)));
        gpuErrchk(cudaMalloc((void **)&d_split_point_list, sizeof(Int) * NActivePoints * Dimension));
        if(use_normal)
            gpuErrchk(cudaMalloc((void **)&d_split_point_ori_index, sizeof(Int) * NActivePoints));

        dIncreGetSplitPointListFlag(d_split_point_list_flag,    
                            d_global_points_blockid,     
                            d_block_input_address_list_size,    
                            d_chunk_points_num,
                            NActivePoints,
                            MAX_INPUT_ADDRESS);

        gpuErrchk(cudaMemcpy(d_split_point_list_flag_long, d_split_point_list_flag, sizeof(uint32_t) * NActivePoints, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_split_point_list_flag_long + NActivePoints, d_split_point_list_flag, sizeof(uint32_t) * NActivePoints, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(d_split_point_list_flag_long + NActivePoints * 2, d_split_point_list_flag, sizeof(uint32_t) * NActivePoints, cudaMemcpyDeviceToDevice));
        
        CUDPP_Compacting compacting(NActivePoints * Dimension);
        
        compacting.apply(d_split_point_list, d_split_point_list_size, d_global_points, d_split_point_list_flag_long, NActivePoints * Dimension);
        
        gpuErrchk(cudaMemcpy(&split_point_list_size, d_split_point_list_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        
        Int return_val = (Int)split_point_list_size / 3;
        
        if(use_normal)
            compacting.apply(d_split_point_ori_index, d_split_point_list_size, d_global_points_ori_index, d_split_point_list_flag, NActivePoints);
#ifdef PRINT_CHUNK
        printf("split_point_list_size = %d\n", return_val);
#endif
        return return_val;
    }
};
