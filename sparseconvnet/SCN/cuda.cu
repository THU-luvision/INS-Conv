// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <Metadata/Metadata.h>
#include <cstdint>

#include "CUDA/ActivePooling.cu"
#include "CUDA/AffineReluTrivialConvolution.cu"
#include "CUDA/AveragePooling.cu"
#include "CUDA/BatchNormalization.cu"
#include "CUDA/BatchwiseMultiplicativeDropout.cu"
#include "CUDA/Convolution.cu"
#include "CUDA/Deconvolution.cu"
#include "CUDA/IOLayers.cu"
#include "CUDA/LeakyReLU.cu"
#include "CUDA/MaxPooling.cu"
#include "CUDA/SparseToDense.cu"
#include "CUDA/UnPooling.cu"
// Submanifold
#include "CUDA/CUDPPWrapper.cu"
#include "CUDA/SubmanifoldRules_cuda.cu"

template void ActivePooling_ForwardPass<float>(float *input_features,
					       float *output_features,
					       Int batchSize, Int maxActive,
					       Int nPlanes, Int *rules,
					       bool average);
template void ActivePooling_BackwardPass<float>(float *d_input_features,
						float *d_output_features,
						Int batchSize, Int maxActive,
						Int nPlanes, Int *rules,
						bool average);

template void dAffineReluTrivialConvolution_forward<float>(
    float *inFeatures, float *outFeatures, float *affineWeight,
    float *affineBias, float *convWeight, Int input_nPlanes, Int input_stride,
    Int output_nPlanes, Int output_stride, Int nActive);
template void dAffineReluTrivialConvolution_backward_dW<float>(
    float *inFeatures, float *dInFeatures, float *dOutFeatures,
    float *affineWeight, float *dAffineWeight, float *affineBias,
    float *dAffineBias, float *convWeight, float *dConvWeight,
    Int input_nPlanes, Int input_stride, Int output_nPlanes, Int output_stride,
    Int nActive, bool additiveGrad);

template void cuda_AveragePooling_ForwardPass<float>(
    float *input_features, float *output_features, Int nPlanes,
    Int input_stride, Int output_stride, RuleBook _rules, Int filterVolume);
template void cuda_AveragePooling_BackwardPass<float>(
    float *d_input_features, float *d_output_features, Int nPlanes,
    Int input_stride, Int output_stride, RuleBook _rules, Int filterVolume);

template void Convolution_fp_bias<float>(float *oF, float *b, Int nPlanes,
					 Int nActive);
template void Convolution_bp_bias<float>(float *d_oF, float *d_b,
					 Int nPlanes, Int nActive);
template double dConvolution_forward2<float>(
    float *inFeatures, float *outFeatures, float *w, RuleBook _rules,
    Int input_nPlanes, Int input_stride, Int output_nPlanes, Int output_stride);


template void dConvolution_backward_dW2_chunkbased<float>(float *inFeatures, float *dInFeatures, float *dOutFeatures,
											RBChunkPointerList& new_rbChunkList,RuleBook &_rules,Int inputFeatureSize,
                                            float *w, float *dw, Int input_nPlanes,
                                            Int input_stride, Int output_nPlanes,
                                            Int output_stride);


template double dConvolution_forward2_chunkbased<float>(float *inFeatures, float *outFeatures, float *w,
                     Int outputFeatureNum,
                     RBChunkPointerList& new_rbChunkList, Int input_nPlanes,
                     Int input_stride, Int output_nPlanes,
                     Int output_stride);
template void dConvolution_backward_dW2<float>(
    float *inFeatures, float *dInFeatures, float *dOutFeatures, float *w,
    float *dw, RuleBook _rules, Int input_nPlanes, Int input_stride,
    Int output_nPlanes, Int output_stride);

template double dDeconvolution_forward2<float>(
    float *inFeatures, float *outFeatures, float *w, RuleBook _rules,
    Int input_nPlanes, Int input_stride, Int output_nPlanes, Int output_stride);

template void dDeconvolution_backward_dW2<float>(
    float *inFeatures, float *dInFeatures, float *dOutFeatures, float *w,
    float *dw, RuleBook _rules, Int input_nPlanes, Int input_stride,
    Int output_nPlanes, Int output_stride);

template void InputLayer_fp<float>(float *input_features,
				   float *output_features, Int nRows,
				   Int maxActive, Int nPlanes, Int *rules_cpu,
				   Int *rules_gpu, bool average);
template void InputLayer_bp<float>(float *d_input_features,
				   float *d_output_features, Int nRows,
				   Int maxActive, Int nPlanes, Int *rules_cpu,
				   Int *rules_gpu, bool average);

template void LeakyReLU_fp<float>(float *input_features, float *output_features,
				  Int n, float alpha);
template void LeakyReLU_bp<float>(float *input_features,
				  float *d_input_features,
				  float *output_features, Int n, float alpha);
template void cuda_MaxPooling_ForwardPass<float>(float *input_features,
						 float *output_features,
						 Int nPlanes, Int input_stride,
						 Int output_stride,
						 RuleBook _rules);
template void cuda_MaxPooling_BackwardPass<float>(
    float *input_features, float *d_input_features, float *output_features,
    float *d_output_features, Int nPlanes, Int input_stride, Int output_stride,
    RuleBook _rules);
template void cuda_SparseToDense_ForwardPass<float>(float *input_features,
						    float *output_features,
						    Int nPlanes,
						    Int spatialVolume,
						    RuleBook _rules);
template void cuda_SparseToDense_BackwardPass<float>(float *d_input_features,
						     float *d_output_features,
						     Int nPlanes,
						     Int spatialVolume,
						     RuleBook _rules);
template void cuda_UnPooling_ForwardPass<float>(float *input_features,
						float *output_features,
						Int nPlanes, Int input_stride,
						Int output_stride,
						RuleBook _rules);
template void cuda_UnPooling_BackwardPass<float>(float *d_input_features,
						 float *d_output_features,
						 Int nPlanes, Int input_stride,
						 Int output_stride,
						 RuleBook _rules);

template void bn_f<float>(float *iF, float *oF, Int nPlanes, Int input_stride,
			  Int output_stride, Int nActive, float *saveMean,
			  float *saveInvStd, float *runningMean,
			  float *runningVar, float *weight, float *bias,
			  float eps, float momentum, bool train,
			  float leakiness);
template void bn_b<float>(float *input_features, float *d_input_features,
			  float *output_features, float *d_output_features,
			  Int nPlanes, Int input_stride, Int output_stride,
			  Int nActive, float *saveMean, float *saveInvStd,
			  float *runningMean, float *runningVar, float *weight,
			  float *bias, float *d_weight, float *d_bias,
			  float leakiness);

template void bmd_f<float>(float *input_features, float *output_features,
			   float *noise, Int nActive, Int nPlanes, float alpha);
template void bmd_b<float>(float *input_features, float *d_input_features,
			   float *d_output_features, float *noise, Int nActive,
			   Int nPlanes, float alpha);

/* -------------------newly added----------------------*/
template double dConvolution_incre_forward2_chunkbased<float>(float *inFeatures, float *outFeatures, float *w,
                                        Int outFeatureNum, RBChunkPointerList& new_rbChunkList, Int input_nPlanes,
                                        Int input_stride, Int output_nPlanes, Int output_stride);

template void InputLayer_inc_fp<float>(float *input_features, float *output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average, uint32_t *pre_exist, float *pre_output_feats, Int *pre_rules_cpu, 
                   Int *pre_rules_gpu, Int pre_nRows, Int pre_maxActive);

template void InputLayer_inc_bp<float>(float *d_input_features, float *d_output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average, Int *pre_exist, float *pre_input_feats);

template
void inc_bn_f<float>(float *iF, float *oF, Int nPlanes, Int input_stride, Int output_stride,
          Int nActive, float *saveMean, float *saveInvStd, float *runningMean,
          float *runningVar, float *weight, float *bias, float eps, float momentum, bool train,
          float leakiness, Int *pre_exist, float *pre_output_feats, float *pre_input_feats);

template 
double dDeconvolution_incre_forward2<float>(float *inFeatures, float *outFeatures, float *w,
			       RuleBook _rules, Int input_nPlanes,
			       Int input_stride, Int output_nPlanes,
			       Int output_stride, float * pre_input_feats, Int *pre_exist_input, Int *pre_exist_out);



template 
double dConvolution_incre_forward2<float>(float *inFeatures, float *outFeatures, float *w,
                                         RuleBook &_rules, Int input_nPlanes,
                                        Int input_stride, Int output_nPlanes, Int output_stride);

void dGetNot(Int *a, Int num);

/*--------------------end of newly added----------------------*/



/* Basic Point Hash */
void dGetPointHash(uint32_t* d_points, uint32_t* d_keys, uint32_t* d_index, Int num);
void dGetPointHashQuery(uint32_t* d_points, uint32_t* d_keys, Int num);
void dMultivalHashGetValue(Int* _d_hash_idx, 
                            uint32_t* d_all_values, 
                            uint2* d_query_vals_multivalue,
                            uint32_t* d_results,
                            Int query_size,
							Int hashtable_size);

/* Compacting */
void dCopyUniquePoints(uint32_t* _d_hash_points, uint32_t* d_points, uint32_t* d_index, Int num, Int size);


/* Submanifold */
void dGetPointsWithinInputRegion(const Int *d_points, Int *d_output_points, Int num, Int size, Int Dimension);

void dGenerateRulebook (Int* d_global_query_result,    // size = NActivePoints * volume
	Int* d_out_rules,              // size = volume * (NActivePoints * 2)
	unsigned* d_out_rules_flag,    // size = volume * (NActivePoints * 2)
	Int NActivePoints,
	Int volume,
	Int ctr);
	

void dGenerateIncreRulebook (Int* d_global_query_result,    // size = NActivePoints * volume
                        Int *d_output_query_result,
                        Int* d_out_rules,              // size = volume * (NActivePoints * 2)
                        unsigned* d_out_rules_flag,    // size = volume * (NActivePoints * 2)
                        Int NActivePoints,
                        Int volume,
                        Int ctr);


void dMultivalPointHashtableInsertHelper(
	uint2* d_index_counts,
	uint32_t* d_all_values,
	uint32_t* d_points,
	Int* _d_hash_idx,
	Int* _d_hash_points,
	uint32_t index_counts_size,
	uint32_t all_values_size);


void dGetBlockGrid(Int *d_input_point, Int *d_blockid, Int block_size, Int length);
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
							Int use_normal);
							
void dgetSplitPointListFlag(uint32_t* d_out_flag,    
							Int* d_blockid,     
							Int* d_output_input_address_size,    
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
);
#ifdef NEW_SPTIAL_POINT
void dGenerateSpatialNewPoint (Int* d_prev_all_point,             // size = num_active_point * dim
	Int* d_next_all_point,                            // size = num_active_point * maxi_sizec * dim
	long* d_size,
    long* d_stride,
    long* d_output_spatial_size,
    Int num_active_point,
    long maxi_sizec,
	Int ndim);
#endif
