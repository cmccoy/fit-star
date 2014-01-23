#ifndef PROTOBUFTOOLS_H
#define PROTOBUFTOOLS_H

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>

#include <boost/iterator/iterator_facade.hpp>

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

template<typename T>
class DelimitedProtocolBufferIterator :
    public boost::iterator_facade<DelimitedProtocolBufferIterator<T>, T, boost::single_pass_traversal_tag>
{
public:
    DelimitedProtocolBufferIterator(std::istream& in, const bool isGzipped = true) :
        in_(&in),
        raw_in_(new google::protobuf::io::IstreamInputStream(in_)),
        zip_in_(new google::protobuf::io::GzipInputStream(raw_in_.get())),
        t_(new T())
    {
        if(isGzipped)
            stream_ = zip_in_.get();
        else
            stream_ = raw_in_.get();
        increment();
    }
    DelimitedProtocolBufferIterator() {};

private:
    friend class boost::iterator_core_access;

    void increment()
    {
        t_->Clear();
        google::protobuf::io::CodedInputStream coded_in(stream_);
        uint32_t size = 0;
        bool success = coded_in.ReadVarint32(&size);
        if(!success) {
            in_ = nullptr;
            return;
        }

        std::string s;
        coded_in.ReadString(&s, size);
        success = t_->ParseFromString(s);
        assert(success && "Failed to parse");
    }

    template<typename O>
    bool equal(const DelimitedProtocolBufferIterator<O>& other) const 
    {
        return t_.get() == other.t_.get();
    }

    T& dereference() const { return *t_; }

    std::istream* in_;
    std::shared_ptr<google::protobuf::io::IstreamInputStream> raw_in_;
    std::shared_ptr<google::protobuf::io::GzipInputStream> zip_in_;
    google::protobuf::io::ZeroCopyInputStream* stream_;
    const std::shared_ptr<T> t_;
};

template<typename T>
std::vector<T> loadDelimitedFromStream(std::istream& in, const bool isGzipped = true)
{
    google::protobuf::io::IstreamInputStream raw_in(&in);
    google::protobuf::io::GzipInputStream zip_in(&raw_in);
    std::vector<T> result;
    google::protobuf::io::ZeroCopyInputStream* stream;
    if(isGzipped)
        stream = &zip_in;
    else
        stream = &raw_in;

    while(true) {
        google::protobuf::io::CodedInputStream coded_in(stream);
        uint32_t size = 0;
        bool success = coded_in.ReadVarint32(&size);
        if(!success) break;
        T item;
        std::string s;
        coded_in.ReadString(&s, size);
        success = item.ParseFromString(s);
        assert(success && "Failed to parse");
        result.push_back(std::move(item));
    }
    return result;
}

template<typename T>
void writeDelimitedToStream(std::ostream& out, const std::vector<T>& items, const bool gzip = true)
{
    google::protobuf::io::OstreamOutputStream raw_out(&out);
    google::protobuf::io::GzipOutputStream zip_out(&raw_out);
    google::protobuf::io::CodedOutputStream coded_out(&zip_out);
    google::protobuf::io::ZeroCopyOutputStream* stream;
    if(gzip)
        stream = &zip_out;
    else
        stream = &raw_out;

    for(const T& item : items) {
        google::protobuf::io::CodedOutputStream coded_out(stream);
        coded_out.WriteVarint32(item.ByteSize());
        item.SerializeWithCachedSizes(&coded_out);
    }
}

#endif
