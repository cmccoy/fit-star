// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: src/mutationio.proto

#ifndef PROTOBUF_src_2fmutationio_2eproto__INCLUDED
#define PROTOBUF_src_2fmutationio_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2004000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2004001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_message_reflection.h>
// @@protoc_insertion_point(includes)

namespace mutationio {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_src_2fmutationio_2eproto();
void protobuf_AssignDesc_src_2fmutationio_2eproto();
void protobuf_ShutdownFile_src_2fmutationio_2eproto();

class Partition;
class MutationCount;

// ===================================================================

class Partition : public ::google::protobuf::Message {
 public:
  Partition();
  virtual ~Partition();
  
  Partition(const Partition& from);
  
  inline Partition& operator=(const Partition& from) {
    CopyFrom(from);
    return *this;
  }
  
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }
  
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }
  
  static const ::google::protobuf::Descriptor* descriptor();
  static const Partition& default_instance();
  
  void Swap(Partition* other);
  
  // implements Message ----------------------------------------------
  
  Partition* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const Partition& from);
  void MergeFrom(const Partition& from);
  void Clear();
  bool IsInitialized() const;
  
  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  
  ::google::protobuf::Metadata GetMetadata() const;
  
  // nested types ----------------------------------------------------
  
  // accessors -------------------------------------------------------
  
  // optional string name = 2;
  inline bool has_name() const;
  inline void clear_name();
  static const int kNameFieldNumber = 2;
  inline const ::std::string& name() const;
  inline void set_name(const ::std::string& value);
  inline void set_name(const char* value);
  inline void set_name(const char* value, size_t size);
  inline ::std::string* mutable_name();
  inline ::std::string* release_name();
  
  // repeated uint32 substitution = 1;
  inline int substitution_size() const;
  inline void clear_substitution();
  static const int kSubstitutionFieldNumber = 1;
  inline ::google::protobuf::uint32 substitution(int index) const;
  inline void set_substitution(int index, ::google::protobuf::uint32 value);
  inline void add_substitution(::google::protobuf::uint32 value);
  inline const ::google::protobuf::RepeatedField< ::google::protobuf::uint32 >&
      substitution() const;
  inline ::google::protobuf::RepeatedField< ::google::protobuf::uint32 >*
      mutable_substitution();
  
  // @@protoc_insertion_point(class_scope:mutationio.Partition)
 private:
  inline void set_has_name();
  inline void clear_has_name();
  
  ::google::protobuf::UnknownFieldSet _unknown_fields_;
  
  ::std::string* name_;
  ::google::protobuf::RepeatedField< ::google::protobuf::uint32 > substitution_;
  
  mutable int _cached_size_;
  ::google::protobuf::uint32 _has_bits_[(2 + 31) / 32];
  
  friend void  protobuf_AddDesc_src_2fmutationio_2eproto();
  friend void protobuf_AssignDesc_src_2fmutationio_2eproto();
  friend void protobuf_ShutdownFile_src_2fmutationio_2eproto();
  
  void InitAsDefaultInstance();
  static Partition* default_instance_;
};
// -------------------------------------------------------------------

class MutationCount : public ::google::protobuf::Message {
 public:
  MutationCount();
  virtual ~MutationCount();
  
  MutationCount(const MutationCount& from);
  
  inline MutationCount& operator=(const MutationCount& from) {
    CopyFrom(from);
    return *this;
  }
  
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }
  
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }
  
  static const ::google::protobuf::Descriptor* descriptor();
  static const MutationCount& default_instance();
  
  void Swap(MutationCount* other);
  
  // implements Message ----------------------------------------------
  
  MutationCount* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const MutationCount& from);
  void MergeFrom(const MutationCount& from);
  void Clear();
  bool IsInitialized() const;
  
  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  
  ::google::protobuf::Metadata GetMetadata() const;
  
  // nested types ----------------------------------------------------
  
  // accessors -------------------------------------------------------
  
  // optional string name = 1;
  inline bool has_name() const;
  inline void clear_name();
  static const int kNameFieldNumber = 1;
  inline const ::std::string& name() const;
  inline void set_name(const ::std::string& value);
  inline void set_name(const char* value);
  inline void set_name(const char* value, size_t size);
  inline ::std::string* mutable_name();
  inline ::std::string* release_name();
  
  // optional double distance = 2;
  inline bool has_distance() const;
  inline void clear_distance();
  static const int kDistanceFieldNumber = 2;
  inline double distance() const;
  inline void set_distance(double value);
  
  // repeated .mutationio.Partition partition = 4;
  inline int partition_size() const;
  inline void clear_partition();
  static const int kPartitionFieldNumber = 4;
  inline const ::mutationio::Partition& partition(int index) const;
  inline ::mutationio::Partition* mutable_partition(int index);
  inline ::mutationio::Partition* add_partition();
  inline const ::google::protobuf::RepeatedPtrField< ::mutationio::Partition >&
      partition() const;
  inline ::google::protobuf::RepeatedPtrField< ::mutationio::Partition >*
      mutable_partition();
  
  // @@protoc_insertion_point(class_scope:mutationio.MutationCount)
 private:
  inline void set_has_name();
  inline void clear_has_name();
  inline void set_has_distance();
  inline void clear_has_distance();
  
  ::google::protobuf::UnknownFieldSet _unknown_fields_;
  
  ::std::string* name_;
  double distance_;
  ::google::protobuf::RepeatedPtrField< ::mutationio::Partition > partition_;
  
  mutable int _cached_size_;
  ::google::protobuf::uint32 _has_bits_[(3 + 31) / 32];
  
  friend void  protobuf_AddDesc_src_2fmutationio_2eproto();
  friend void protobuf_AssignDesc_src_2fmutationio_2eproto();
  friend void protobuf_ShutdownFile_src_2fmutationio_2eproto();
  
  void InitAsDefaultInstance();
  static MutationCount* default_instance_;
};
// ===================================================================


// ===================================================================

// Partition

// optional string name = 2;
inline bool Partition::has_name() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void Partition::set_has_name() {
  _has_bits_[0] |= 0x00000001u;
}
inline void Partition::clear_has_name() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void Partition::clear_name() {
  if (name_ != &::google::protobuf::internal::kEmptyString) {
    name_->clear();
  }
  clear_has_name();
}
inline const ::std::string& Partition::name() const {
  return *name_;
}
inline void Partition::set_name(const ::std::string& value) {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  name_->assign(value);
}
inline void Partition::set_name(const char* value) {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  name_->assign(value);
}
inline void Partition::set_name(const char* value, size_t size) {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  name_->assign(reinterpret_cast<const char*>(value), size);
}
inline ::std::string* Partition::mutable_name() {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  return name_;
}
inline ::std::string* Partition::release_name() {
  clear_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    return NULL;
  } else {
    ::std::string* temp = name_;
    name_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
    return temp;
  }
}

// repeated uint32 substitution = 1;
inline int Partition::substitution_size() const {
  return substitution_.size();
}
inline void Partition::clear_substitution() {
  substitution_.Clear();
}
inline ::google::protobuf::uint32 Partition::substitution(int index) const {
  return substitution_.Get(index);
}
inline void Partition::set_substitution(int index, ::google::protobuf::uint32 value) {
  substitution_.Set(index, value);
}
inline void Partition::add_substitution(::google::protobuf::uint32 value) {
  substitution_.Add(value);
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::uint32 >&
Partition::substitution() const {
  return substitution_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::uint32 >*
Partition::mutable_substitution() {
  return &substitution_;
}

// -------------------------------------------------------------------

// MutationCount

// optional string name = 1;
inline bool MutationCount::has_name() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void MutationCount::set_has_name() {
  _has_bits_[0] |= 0x00000001u;
}
inline void MutationCount::clear_has_name() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void MutationCount::clear_name() {
  if (name_ != &::google::protobuf::internal::kEmptyString) {
    name_->clear();
  }
  clear_has_name();
}
inline const ::std::string& MutationCount::name() const {
  return *name_;
}
inline void MutationCount::set_name(const ::std::string& value) {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  name_->assign(value);
}
inline void MutationCount::set_name(const char* value) {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  name_->assign(value);
}
inline void MutationCount::set_name(const char* value, size_t size) {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  name_->assign(reinterpret_cast<const char*>(value), size);
}
inline ::std::string* MutationCount::mutable_name() {
  set_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    name_ = new ::std::string;
  }
  return name_;
}
inline ::std::string* MutationCount::release_name() {
  clear_has_name();
  if (name_ == &::google::protobuf::internal::kEmptyString) {
    return NULL;
  } else {
    ::std::string* temp = name_;
    name_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
    return temp;
  }
}

// optional double distance = 2;
inline bool MutationCount::has_distance() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void MutationCount::set_has_distance() {
  _has_bits_[0] |= 0x00000002u;
}
inline void MutationCount::clear_has_distance() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void MutationCount::clear_distance() {
  distance_ = 0;
  clear_has_distance();
}
inline double MutationCount::distance() const {
  return distance_;
}
inline void MutationCount::set_distance(double value) {
  set_has_distance();
  distance_ = value;
}

// repeated .mutationio.Partition partition = 4;
inline int MutationCount::partition_size() const {
  return partition_.size();
}
inline void MutationCount::clear_partition() {
  partition_.Clear();
}
inline const ::mutationio::Partition& MutationCount::partition(int index) const {
  return partition_.Get(index);
}
inline ::mutationio::Partition* MutationCount::mutable_partition(int index) {
  return partition_.Mutable(index);
}
inline ::mutationio::Partition* MutationCount::add_partition() {
  return partition_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::mutationio::Partition >&
MutationCount::partition() const {
  return partition_;
}
inline ::google::protobuf::RepeatedPtrField< ::mutationio::Partition >*
MutationCount::mutable_partition() {
  return &partition_;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace mutationio

#ifndef SWIG
namespace google {
namespace protobuf {


}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_src_2fmutationio_2eproto__INCLUDED
