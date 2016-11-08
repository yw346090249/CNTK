//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma  once

#include "CNTKLibrary.h"

namespace CNTK
{
    ///
    /// Distributed Trainer.
    ///
    class DataParallelDistributedTrainer : public DistributedTrainer
    {
    public:
        DataParallelDistributedTrainer(DistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate);

        // Optional override that gets called before each minbatch during training
        void PreMinibatchCallback(const Trainer& trainer) override;

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        bool PreParameterUpdateCallback(const Trainer& trainer, std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues, MinibatchInfo& info) override;

        // Optionally overridable method to get checkpoint state associated with this Distributed train method
        Dictionary CreateCheckpoint(const Trainer& trainer, const Dictionary& localStateToShare) override;

        // Optionally overridable method to restore state pertaining this distributed training method from a previous checkpoint
        Dictionary RestoreFromCheckpoint(const Dictionary& checkpoint) override;

    private:
        DistributedCommunicatorPtr GetCommunicator() override
        {
            return m_communicator;
        }

        DistributedCommunicatorPtr m_communicator;
        bool m_useAsyncBufferedParameterUpdate;
    };
}