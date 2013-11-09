// Generated by the protocol buffer compiler.  DO NOT EDIT!

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "mutationlist.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace mutationio {

namespace {

const ::google::protobuf::Descriptor* MutationCount_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  MutationCount_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_mutationlist_2eproto() {
  protobuf_AddDesc_mutationlist_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "mutationlist.proto");
  GOOGLE_CHECK(file != NULL);
  MutationCount_descriptor_ = file->message_type(0);
  static const int MutationCount_offsets_[3] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MutationCount, name_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MutationCount, distance_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MutationCount, mutations_),
  };
  MutationCount_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      MutationCount_descriptor_,
      MutationCount::default_instance_,
      MutationCount_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MutationCount, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(MutationCount, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(MutationCount));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_mutationlist_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    MutationCount_descriptor_, &MutationCount::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_mutationlist_2eproto() {
  delete MutationCount::default_instance_;
  delete MutationCount_reflection_;
}

void protobuf_AddDesc_mutationlist_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\022mutationlist.proto\022\nmutationio\"B\n\rMuta"
    "tionCount\022\014\n\004name\030\001 \001(\t\022\020\n\010distance\030\002 \001("
    "\001\022\021\n\tmutations\030\003 \003(\r", 100);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "mutationlist.proto", &protobuf_RegisterTypes);
  MutationCount::default_instance_ = new MutationCount();
  MutationCount::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_mutationlist_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_mutationlist_2eproto {
  StaticDescriptorInitializer_mutationlist_2eproto() {
    protobuf_AddDesc_mutationlist_2eproto();
  }
} static_descriptor_initializer_mutationlist_2eproto_;


// ===================================================================

#ifndef _MSC_VER
const int MutationCount::kNameFieldNumber;
const int MutationCount::kDistanceFieldNumber;
const int MutationCount::kMutationsFieldNumber;
#endif  // !_MSC_VER

MutationCount::MutationCount()
  : ::google::protobuf::Message() {
  SharedCtor();
}

void MutationCount::InitAsDefaultInstance() {
}

MutationCount::MutationCount(const MutationCount& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
}

void MutationCount::SharedCtor() {
  _cached_size_ = 0;
  name_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
  distance_ = 0;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

MutationCount::~MutationCount() {
  SharedDtor();
}

void MutationCount::SharedDtor() {
  if (name_ != &::google::protobuf::internal::kEmptyString) {
    delete name_;
  }
  if (this != default_instance_) {
  }
}

void MutationCount::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* MutationCount::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return MutationCount_descriptor_;
}

const MutationCount& MutationCount::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_mutationlist_2eproto();  return *default_instance_;
}

MutationCount* MutationCount::default_instance_ = NULL;

MutationCount* MutationCount::New() const {
  return new MutationCount;
}

void MutationCount::Clear() {
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (has_name()) {
      if (name_ != &::google::protobuf::internal::kEmptyString) {
        name_->clear();
      }
    }
    distance_ = 0;
  }
  mutations_.Clear();
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool MutationCount::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) return false
  ::google::protobuf::uint32 tag;
  while ((tag = input->ReadTag()) != 0) {
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional string name = 1;
      case 1: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_name()));
          ::google::protobuf::internal::WireFormat::VerifyUTF8String(
            this->name().data(), this->name().length(),
            ::google::protobuf::internal::WireFormat::PARSE);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(17)) goto parse_distance;
        break;
      }
      
      // optional double distance = 2;
      case 2: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_FIXED64) {
         parse_distance:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 input, &distance_)));
          set_has_distance();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(24)) goto parse_mutations;
        break;
      }
      
      // repeated uint32 mutations = 3;
      case 3: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_mutations:
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 1, 24, input, this->mutable_mutations())));
        } else if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag)
                   == ::google::protobuf::internal::WireFormatLite::
                      WIRETYPE_LENGTH_DELIMITED) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitiveNoInline<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, this->mutable_mutations())));
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(24)) goto parse_mutations;
        if (input->ExpectAtEnd()) return true;
        break;
      }
      
      default: {
      handle_uninterpreted:
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          return true;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
  return true;
#undef DO_
}

void MutationCount::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // optional string name = 1;
  if (has_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->name().data(), this->name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    ::google::protobuf::internal::WireFormatLite::WriteString(
      1, this->name(), output);
  }
  
  // optional double distance = 2;
  if (has_distance()) {
    ::google::protobuf::internal::WireFormatLite::WriteDouble(2, this->distance(), output);
  }
  
  // repeated uint32 mutations = 3;
  for (int i = 0; i < this->mutations_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(
      3, this->mutations(i), output);
  }
  
  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
}

::google::protobuf::uint8* MutationCount::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // optional string name = 1;
  if (has_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->name().data(), this->name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->name(), target);
  }
  
  // optional double distance = 2;
  if (has_distance()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteDoubleToArray(2, this->distance(), target);
  }
  
  // repeated uint32 mutations = 3;
  for (int i = 0; i < this->mutations_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteUInt32ToArray(3, this->mutations(i), target);
  }
  
  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  return target;
}

int MutationCount::ByteSize() const {
  int total_size = 0;
  
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // optional string name = 1;
    if (has_name()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::StringSize(
          this->name());
    }
    
    // optional double distance = 2;
    if (has_distance()) {
      total_size += 1 + 8;
    }
    
  }
  // repeated uint32 mutations = 3;
  {
    int data_size = 0;
    for (int i = 0; i < this->mutations_size(); i++) {
      data_size += ::google::protobuf::internal::WireFormatLite::
        UInt32Size(this->mutations(i));
    }
    total_size += 1 * this->mutations_size() + data_size;
  }
  
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void MutationCount::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const MutationCount* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const MutationCount*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void MutationCount::MergeFrom(const MutationCount& from) {
  GOOGLE_CHECK_NE(&from, this);
  mutations_.MergeFrom(from.mutations_);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_name()) {
      set_name(from.name());
    }
    if (from.has_distance()) {
      set_distance(from.distance());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void MutationCount::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void MutationCount::CopyFrom(const MutationCount& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool MutationCount::IsInitialized() const {
  
  return true;
}

void MutationCount::Swap(MutationCount* other) {
  if (other != this) {
    std::swap(name_, other->name_);
    std::swap(distance_, other->distance_);
    mutations_.Swap(&other->mutations_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata MutationCount::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = MutationCount_descriptor_;
  metadata.reflection = MutationCount_reflection_;
  return metadata;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace mutationio

// @@protoc_insertion_point(global_scope)