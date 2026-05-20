/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cacheTransBuffer.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/executor/executor.h"

#include <NvInferRuntimeBase.h>
#include <algorithm>
#include <mutex>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

// ============================================================================
// FabricMemory Implementation
// ============================================================================

class FabricMemory::Impl
{
public:
    Impl(size_t size)
        : mSize(size)
    {
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceIdx));
        CUmemAllocationHandleType const handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
        CUmemAllocationProp prop = {};
        prop.requestedHandleTypes = handle_type;
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = mDeviceIdx;
        prop.allocFlags.gpuDirectRDMACapable = 1;

        size_t granularity{0};
        TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        mGranularity = granularity;
        mAllocSize = (size + granularity - 1) / granularity * granularity;
        TLLM_CU_CHECK(cuMemCreate(&mHandle, mAllocSize, &prop, 0));
        TLLM_CU_CHECK(cuMemAddressReserve(&mDevicePtr, mAllocSize, mGranularity, 0, 0));
        mPtr = reinterpret_cast<void*>(mDevicePtr);
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        accessDesc.location.id = mDeviceIdx;
        TLLM_CU_CHECK(cuMemMap(mDevicePtr, mAllocSize, 0, mHandle, 0));
        TLLM_CU_CHECK(cuMemSetAccess(mDevicePtr, mAllocSize, &accessDesc, 1));
        TLLM_LOG_DEBUG("FabricMemory::Impl::Impl mAllocSize:%ld", mAllocSize);
    }

    ~Impl()
    {
        TLLM_LOG_DEBUG("FabricMemory::Impl::~Impl mAllocSize:%ld", mAllocSize);
        TLLM_CU_CHECK(cuMemUnmap(mDevicePtr, mAllocSize));
        TLLM_CU_CHECK(cuMemRelease(mHandle));
        TLLM_CU_CHECK(cuMemAddressFree(mDevicePtr, mAllocSize));
    }

    [[nodiscard]] void* getPtr() const
    {
        return mPtr;
    }

    [[nodiscard]] size_t getSize() const
    {
        return mSize;
    }

private:
    size_t mSize;
    size_t mAllocSize;
    size_t mGranularity;
    void* mPtr;
    CUdeviceptr mDevicePtr;
    CUmemGenericAllocationHandle mHandle;
    int mDeviceIdx;
};

FabricMemory::FabricMemory(size_t size)
    : pImpl(std::make_unique<Impl>(size))
{
}

FabricMemory::~FabricMemory() = default;

FabricMemory::FabricMemory(FabricMemory&&) noexcept = default;
FabricMemory& FabricMemory::operator=(FabricMemory&&) noexcept = default;

void* FabricMemory::getPtr() const
{
    return pImpl->getPtr();
}

size_t FabricMemory::getSize() const
{
    return pImpl->getSize();
}

size_t FabricMemory::getAlignedSize(size_t size)
{
    int deviceIdx = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceIdx));
    CUmemAllocationHandleType const handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemAllocationProp prop = {};
    prop.requestedHandleTypes = handle_type;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceIdx;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t granularity{0};
    TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    return (size + granularity - 1) / granularity * granularity;
}

bool FabricMemory::supportFbaricMemory()
{
#ifdef __aarch64__
    auto support_fun = []()
    {
        int fabric_handle_supported{0};
        int gpu_direct_rdma_with_cuda_vmm_supported{0};
        int deviceIdx = 0;
        TLLM_CUDA_CHECK(cudaGetDevice(&deviceIdx));
        CUresult ret0 = cuDeviceGetAttribute(
            &fabric_handle_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, deviceIdx);

        CUresult ret1 = cuDeviceGetAttribute(&gpu_direct_rdma_with_cuda_vmm_supported,
            CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, deviceIdx);
        TLLM_LOG_DEBUG("FabricMemory::supportFabricMemory fabric_handle_supported:%d", fabric_handle_supported);
        TLLM_LOG_DEBUG("FabricMemory::supportFabricMemory gpu_direct_rdma_with_cuda_vmm_supported:%d",
            gpu_direct_rdma_with_cuda_vmm_supported);
        if (ret0 != CUresult::CUDA_SUCCESS || ret1 != CUresult::CUDA_SUCCESS || fabric_handle_supported == 0
            || gpu_direct_rdma_with_cuda_vmm_supported == 0)
        {
            return false;
        }

        CUmemAllocationHandleType const handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
        CUmemAllocationProp prop = {};
        prop.requestedHandleTypes = handle_type;
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = deviceIdx;
        prop.allocFlags.gpuDirectRDMACapable = 1;

        size_t granularity{0};
        TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        CUmemGenericAllocationHandle handle;

        auto cuRet = cuMemCreate(&handle, granularity, &prop, 0);

        if (cuRet == CUresult::CUDA_SUCCESS)
        {
            TLLM_CU_CHECK(cuMemRelease(handle));
            return true;
        }
        if (cuRet == CUresult::CUDA_ERROR_NOT_PERMITTED)
        {
            TLLM_LOG_WARNING("Try to creat fabric memory failed , setting imex channel may be required");
            return false;
        }
        TLLM_CU_CHECK(cuRet);

        return false;
    };
    static bool support = support_fun();
    return support;

#else
    return false;
#endif
}

// ============================================================================
// CacheTransBufferManager Implementation
// ============================================================================

size_t CacheTransBufferManager::computeRnnPoolBufferSize(KVCacheManager::LinearAttentionMetadata const& linearMeta,
    KVCacheManager::BlockManager const& blockManager, SizeType32 numLocalMambaLayers)
{
    SizeType32 const tokensPerBlock = blockManager.getTokensPerBlock();
    SizeType32 const maxBlocksPerSeq = blockManager.getWindowSizeMetadata(linearMeta.cacheType).maxBlocksPerSeq;

    SizeType32 maxRealBlocks = 1; // At minimum, the final state block
    if (linearMeta.statesSnapshotInterval > 0)
    {
        SizeType32 const snapshotBlockInterval = std::max(
            static_cast<SizeType32>(1), static_cast<SizeType32>(linearMeta.statesSnapshotInterval / tokensPerBlock));
        maxRealBlocks = maxBlocksPerSeq / snapshotBlockInterval + 1;
        if (linearMeta.saveLastSnapshot)
        {
            maxRealBlocks += 1;
        }
    }
    return static_cast<size_t>(maxRealBlocks) * numLocalMambaLayers * linearMeta.allRecurrentStatesBytes;
}

size_t CacheTransBufferManager::computeTransferBufferSize(
    KVCacheManager::BaseKVCacheManager* cacheManager, std::optional<size_t> maxNumTokens, bool transferIndexerKCache)
{
    nvinfer1::DataType dataType;
    if (transferIndexerKCache)
    {
        dataType = cacheManager->getIndexerKCachePool()->getDataType();
    }
    else
    {
        dataType = cacheManager->getPrimaryPool(0)->getDataType();
    }

    auto tokensPerBlock = cacheManager->getBlockManager().getTokensPerBlock();
    size_t bufferSizeFromMaxNumToken = 0;

    if (maxNumTokens.has_value())
    {
        TLLM_CHECK(maxNumTokens.value() % tokensPerBlock == 0);
        auto dataSize = common::getDTypeSize(dataType);
        SizeType32 kvCacheByteSizePerTokenPerLayer = 0;
        if (transferIndexerKCache)
        {
            kvCacheByteSizePerTokenPerLayer
                = cacheManager->getIndexerKCachePool()->getDimension<-1>() * dataSize / tokensPerBlock;
        }
        else
        {
            auto primaryPool = cacheManager->getPrimaryPool(0);
            kvCacheByteSizePerTokenPerLayer
                = primaryPool->getDimension<-1>() * primaryPool->getDimension<2>() * dataSize / tokensPerBlock;
        }
        for (auto layerId = 0; layerId < cacheManager->getBlockManager().getNumLayers(); layerId++)
        {
            auto poolIdx = cacheManager->getBlockManager().getLayerPoolIdx(layerId);
            auto windowSize = static_cast<size_t>(cacheManager->getBlockManager().getPoolWindowSize(poolIdx));
            auto alignedWindowSize = (windowSize + tokensPerBlock - 1) / tokensPerBlock * tokensPerBlock;
            auto validTokenNum = (alignedWindowSize < maxNumTokens.value() ? alignedWindowSize : maxNumTokens.value());
            if (common::getEnvKVCacheTransferAllBlocksForWindow())
            {
                validTokenNum = maxNumTokens.value();
            }
            validTokenNum += tokensPerBlock; // add one more block

            bufferSizeFromMaxNumToken += validTokenNum * kvCacheByteSizePerTokenPerLayer;
        }
    }

    size_t kvSize
        = maxNumTokens.has_value() ? bufferSizeFromMaxNumToken : common::getEnvMemSizeForKVCacheTransferBuffer();

    // For unified pool hybrid models (CppMambaHybridCacheManager), also consider
    // the RNN state transfer size so the buffer can serve both KV and RNN sequentially.
    // Only inflate the primary KV buffer, not the MLA indexer buffer.
    auto const& blockManager = cacheManager->getBlockManager();
    auto const& linearMeta = blockManager.getLinearAttentionMetadata();
    if (linearMeta.has_value() && !transferIndexerKCache)
    {
        // Count local mamba layers from block manager (layers using recurrent state pool).
        SizeType32 numLocalMambaLayers = 0;
        for (SizeType32 layerId = 0; layerId < blockManager.getNumLayers(); ++layerId)
        {
            auto poolIdx = blockManager.getLayerPoolIdx(layerId);
            auto ws = blockManager.getPoolWindowSize(poolIdx);
            if (LinearAttentionMetadata::hasRecurrentStatesCache(ws))
            {
                numLocalMambaLayers++;
            }
        }
        if (numLocalMambaLayers > 0)
        {
            size_t rnnSize = computeRnnPoolBufferSize(*linearMeta, blockManager, numLocalMambaLayers);
            kvSize = std::max(kvSize, rnnSize);
        }
    }

    return kvSize;
}

CacheTransBufferManager::CacheTransBufferManager(
    KVCacheManager::BaseKVCacheManager* cacheManager, std::optional<size_t> maxNumTokens, bool transferIndexerKCache)
    : BaseTransBufferManager(computeTransferBufferSize(cacheManager, maxNumTokens, transferIndexerKCache),
        transferIndexerKCache ? cacheManager->getIndexerKCachePool()->getDataType()
                              : cacheManager->getPrimaryPool(0)->getDataType(),
        maxNumTokens)
    , mCacheManager{cacheManager}
    , mTransferIndexerKCache{transferIndexerKCache}
{
    // TODO: FP4 dataSize
    TLLM_CHECK(mCacheManager);
    TLLM_LOG_INFO("CacheTransBufferManager created for KV cache");
}

size_t CacheTransBufferManager::preAllocBufferSize(
    std::map<SizeType32, SizeType32> const& cacheSizeBytesPerTokenPerWindow, SizeType32 tokensPerBlock,
    std::optional<executor::CacheTransceiverConfig> const& cacheTransceiverConfig)
{
    if (!cacheTransceiverConfig.has_value())
    {
        return 0;
    }
    if (!cacheTransceiverConfig->getBackendType().has_value())
    {
        return 0;
    }
    auto maxNumTokens = cacheTransceiverConfig->getMaxTokensInBuffer();
    size_t transferBufferSize = common::getEnvMemSizeForKVCacheTransferBuffer();
    if (maxNumTokens.has_value())
    {
        transferBufferSize = 0;
        for (auto const& [windowSize, cacheSizeBytesPerToken] : cacheSizeBytesPerTokenPerWindow)
        {
            auto alignedWindowSize = (windowSize + tokensPerBlock - 1) / tokensPerBlock * tokensPerBlock;
            auto validTokenNum = (static_cast<size_t>(alignedWindowSize) < maxNumTokens.value()
                    ? static_cast<size_t>(alignedWindowSize)
                    : maxNumTokens.value());
            if (common::getEnvKVCacheTransferAllBlocksForWindow())
            {
                validTokenNum = maxNumTokens.value();
            }
            validTokenNum += tokensPerBlock; // add one more block
            transferBufferSize += validTokenNum * cacheSizeBytesPerToken;
        }
    }
    bool useFabricMemory = FabricMemory::supportFbaricMemory()
        && (!(common::getEnvKVCacheTransferUseSyncBuffer() || common::getEnvKVCacheTransferUseAsyncBuffer()));
    if (useFabricMemory)
    {
        transferBufferSize = FabricMemory::getAlignedSize(transferBufferSize);
    }
    size_t recvBufferCount = common::getEnvRequestKVCacheConcurrent() ? common::getEnvKVCacheRecvBufferCount() : 1;
    size_t sendBufferCount = common::getEnvKVCacheSendMaxConcurrenceNum();
    size_t preAllocBufferSize = transferBufferSize * (recvBufferCount + sendBufferCount);
    return preAllocBufferSize;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
