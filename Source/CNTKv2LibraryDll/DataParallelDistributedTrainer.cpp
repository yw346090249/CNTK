//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "DataParallelDistributedTrainer.h"
#include "DistributedCommunicator.h"

#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
#include "QuantizedDistributedCommunicator.h"
#include "QuantizedDataParallelDistributedTrainer.h"
#include "BlockMomentumDistributedTrainer.h"
#endif

namespace CNTK
{
#ifdef CNTK_PARALLEL_TRAINING_SUPPORT
    QuantizedDistributedCommunicatorPtr QuantizedMPICommunicator(bool zeroThresholdFor1Bit, bool useQuantizationForSelfStripe, size_t numQuantizationBits)
    {
        return MakeSharedObject<QuantizedMPICommunicatorImpl>(zeroThresholdFor1Bit, useQuantizationForSelfStripe, numQuantizationBits);
    }

    DistributedTrainerPtr CreateQuantizedDataParallelDistributedTrainer(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate)
    {
        return MakeSharedObject<DataParallelDistributedTrainer>(communicator, useAsyncBufferedParameterUpdate);
    }

    DistributedTrainerPtr CreateBlockMomentumDistributedTrainer(
        DistributedCommunicatorPtr communicator,
        size_t blockSize,
        bool useNestrovMomentum,
        bool resetSGDMomentumAfterAggregation,
        double blockLearningRate)
    {
        return MakeSharedObject<BlockMomentumDistributedTrainer>(
            communicator,
            blockSize,
            useNestrovMomentum,
            resetSGDMomentumAfterAggregation,
            blockLearningRate);
    }

    DistributedTrainerPtr CreateBlockMomentumDistributedTrainer(
        DistributedCommunicatorPtr communicator,
        size_t blockSize,
        double blockMomentumAsTimeConstant,
        bool useNestrovMomentum,
        bool resetSGDMomentumAfterAggregation,
        double blockLearningRate)
    {
        return MakeSharedObject<BlockMomentumDistributedTrainer>(
            communicator,
            blockSize,
            useNestrovMomentum,
            resetSGDMomentumAfterAggregation,
            blockLearningRate,
            blockMomentumAsTimeConstant);
    }

#else
    QuantizedDistributedCommunicatorPtr QuantizedMPICommunicator(bool, bool, size_t)
    {
        LogicError("Quantized MPI Communicator is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedTrainerPtr CreateQuantizedDataParallelDistributedTrainer(DistributedCommunicatorPtr, bool)
    {
        LogicError("Quantized Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedTrainerPtr CreateBlockMomentumDistributedTrainer(
        DistributedCommunicatorPtr /*communicator*/,
        size_t /*blockSize*/,
        bool /*useNestrovMomentum*/,
        bool /*resetSGDMomentumAfterAggregation*/,
        double /*blockLearningRate*/)
    {
        LogicError("Block Momentum Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }

    DistributedTrainerPtr CreateBlockMomentumDistributedTrainer(
        DistributedCommunicatorPtr /*communicator*/,
        size_t /*blockSize*/,
        double /*blockMomentumAsTimeConstant*/,
        bool /*useNestrovMomentum*/,
        bool /*resetSGDMomentumAfterAggregation*/,
        double /*blockLearningRate*/)
    {
        LogicError("Block Momentum Distributed Trainer is not supported for this build. The 1BitSGD build is needed, see CNTK wiki for details.");
    }
#endif

    DistributedTrainerPtr CreateDataParallelDistributedTrainer(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate)
    {
        return MakeSharedObject<DataParallelDistributedTrainer>(communicator, useAsyncBufferedParameterUpdate);
    }

    DataParallelDistributedTrainer::DataParallelDistributedTrainer(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate)
        : m_communicator(communicator),
        m_useAsyncBufferedParameterUpdate(useAsyncBufferedParameterUpdate)
    {
        if (useAsyncBufferedParameterUpdate)
            LogicError("Asynchronous parameter update is not yet supported.");
    }

    // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
    bool DataParallelDistributedTrainer::PreParameterUpdateCallback(const Trainer& /*trainer*/, std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues, MinibatchInfo& info)
    {
        std::vector<NDArrayViewPtr> valuesToAggregate;
        for (const auto& i : gradientValues)
            valuesToAggregate.push_back(i.second);
        valuesToAggregate.push_back(info.evalCriterionValue);
        valuesToAggregate.push_back(info.trainingLossValue);

        auto value = MakeSharedObject<NDArrayView>(static_cast<double>(info.numberOfSamples), NDShape{1}, DeviceDescriptor::CPUDevice());
        valuesToAggregate.push_back(value);

        m_communicator->AggregateInPlace(valuesToAggregate, m_communicator->Workers());

        info.numberOfSamples = static_cast<size_t>(*valuesToAggregate.back()->WritableDataBuffer<double>());
        return info.numberOfSamples == 0;
    }

    // Optional override that gets called before each minbatch during training
    void DataParallelDistributedTrainer::PreMinibatchCallback(const Trainer& /*trainer*/)
    {
    }

    // Optionally overridable method to get checkpoint state associated with this Distributed train method
    Dictionary DataParallelDistributedTrainer::CreateCheckpoint(const Trainer&, const Dictionary& localStateToShare)
    {
        std::vector<DictionaryPtr> remoteState;
        m_communicator->Gather(localStateToShare, remoteState, m_communicator->Workers());

        Dictionary result;
        for (size_t i = 0; i < m_communicator->Workers().size(); ++i)
        {
            result[std::to_wstring(i)] = *remoteState[i];
        }

        return result;
    }

    // Optionally overridable method to restore state pertaining this distributed training method from a previous checkpoint
    Dictionary DataParallelDistributedTrainer::RestoreFromCheckpoint(const Dictionary& checkpoint)
    {
        auto key = std::to_wstring(m_communicator->CurrentWorker().m_globalRank);
        if (checkpoint.Contains(key))
            return checkpoint[key].Value<Dictionary>();

        // Return 0 rank if possible.
        key = std::to_wstring(0);
        if (!checkpoint.Contains(key))
            RuntimeError("Cannot restore from the checkpoint, 0 rank is missing.");
        return checkpoint[key].Value<Dictionary>();
    }
}
