//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "StringToIdMap.h"
#include <set>
#include <functional>
#include <sstream>

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents a full corpus.
// Defines which sequences should participate in the reading.
// TODO: Extract an interface.
class CorpusDescriptor
{
public:
    CorpusDescriptor(const std::wstring& file, bool numericSequenceKeys) : CorpusDescriptor(numericSequenceKeys)
    {
        m_includeAll = false;

        // Add all sequence ids.
        for (msra::files::textreader r(file); r;)
        {
            m_sequenceIds.insert(KeyToId(r.getline()));
        }
    }

    // By default include all sequences.
    CorpusDescriptor(bool numericSequenceKeys) : m_includeAll(true), m_numericSequenceKeys(numericSequenceKeys)
    {
        if (numericSequenceKeys)
        {
            KeyToId = [](const std::string& key)
            {
                size_t id;
                int converted = sscanf_s(key.c_str(), "%llu", &id);
                if (converted != key.size())
                    RuntimeError("Invalid numeric sequence id %s", key.c_str());
                return id;
            };

            IdToKey = [](size_t id)
            {
                return std::to_string(id);
            };
        }
        else
        {
            KeyToId = [this](const std::string& key)
            {
                size_t id;
                if (m_stringRegistry.TryGet(key, id))
                    return id;
                return m_stringRegistry.AddValue(key);
            };

            IdToKey = [this](size_t id)
            {
                return m_stringRegistry[id];
            };
        }
    }

    // Checks if the specified sequence key should be used for reading.
    bool IsIncluded(const std::string& sequenceKey)
    {
        if (m_includeAll)
        {
            return true;
        }

        size_t id = 0;
        if (m_numericSequenceKeys)
            id = KeyToId(sequenceKey);
        else
        {
            if (!m_stringRegistry.TryGet(sequenceKey, id))
                return false;
        }
        return m_sequenceIds.find(id) != m_sequenceIds.end();
    }

    std::function<size_t(const std::string&)> KeyToId;
    std::function<std::string(size_t)> IdToKey;

private:
    DISABLE_COPY_AND_MOVE(CorpusDescriptor);
    bool m_numericSequenceKeys;
    bool m_includeAll;
    std::set<size_t> m_sequenceIds;

    StringToIdMap m_stringRegistry;
};

typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

}}}
