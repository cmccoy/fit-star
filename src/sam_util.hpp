#ifndef SAM_UTIL_H
#define SAM_UTIL_H

#include "faidx.h"
#include "sam.h"

#include <boost/iterator/iterator_facade.hpp>

struct SamRecord {
    SamRecord() : record(bam_init1()) {}
    ~SamRecord() { if(record != nullptr) bam_destroy1(record); }

    SamRecord(const SamRecord& other) = delete;
    SamRecord(SamRecord&& other) :
        record(other.record)
    {
        other.record = nullptr;
    }

    SamRecord& operator=(const SamRecord& other) = delete;
    SamRecord& operator=(SamRecord&& other)
    {
        record = other.record;
        other.record = nullptr;
        return *this;
    }

    bam1_t* record;
};

struct SamFile {
    SamFile(const std::string& path, const std::string& mode = "rb", void* extra = nullptr) :
        fp(samopen(path.c_str(), mode.c_str(), extra))
    {
        assert(fp != nullptr && "Failed to open BAM");
    }

    SamFile(const SamFile& other) = delete;
    SamFile& operator=(const SamFile& other) = delete;


    ~SamFile()
    {
        if(fp != nullptr)
            samclose(fp);
    }

    samfile_t* fp;
};

struct FastaIndex {
    FastaIndex(const std::string& path) :
        index(fai_load(path.c_str())) {}
    ~FastaIndex() { if(index != nullptr) fai_destroy(index); }
    faidx_t* index;
};

/// An iterator over a BAM file
class SamIterator :
    public boost::iterator_facade<SamIterator, bam1_t, boost::single_pass_traversal_tag, bam1_t*>
{
public:
    SamIterator() : bamrec_(nullptr) {}

    /// Constructor
    ///
    /// \param fp
    /// \param bamrec Initialized BAM record - is immediately filled with first entry from fp
    SamIterator(samfile_t* fp, bam1_t* bamrec) : fp_(fp), bamrec_(bamrec) {
        increment();
    }

private:
    friend class boost::iterator_core_access;

    void increment()
    {
        int result = samread(fp_, bamrec_);
        if(result <= 0) {
            bamrec_ = nullptr;
        }
    }

    bool equal(const SamIterator & other) const
    {
        return this->bamrec_ == other.bamrec_;
    }

    bam1_t* dereference() const { return bamrec_; }

    samfile_t* fp_;
    bam1_t* bamrec_;
};

#endif
