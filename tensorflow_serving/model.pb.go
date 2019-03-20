// Code generated by protoc-gen-go. DO NOT EDIT.
// source: model.proto

package tensorflow_serving

import (
	fmt "fmt"
	math "math"

	proto "github.com/golang/protobuf/proto"
	wrappers "github.com/golang/protobuf/ptypes/wrappers"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion2 // please upgrade the proto package

// Metadata for an inference request such as the model name and version.
type ModelSpec struct {
	// Required servable name.
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// Optional version. If unspecified, will use the latest (numerical) version.
	// Typically not needed unless coordinating across multiple models that were
	// co-trained and/or have inter-dependencies on the versions used at inference
	// time.
	Version *wrappers.Int64Value `protobuf:"bytes,2,opt,name=version,proto3" json:"version,omitempty"`
	// A named signature to evaluate. If unspecified, the default signature will
	// be used. Note that only MultiInference will initially support this.
	SignatureName        string   `protobuf:"bytes,3,opt,name=signature_name,json=signatureName,proto3" json:"signature_name,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ModelSpec) Reset()         { *m = ModelSpec{} }
func (m *ModelSpec) String() string { return proto.CompactTextString(m) }
func (*ModelSpec) ProtoMessage()    {}
func (*ModelSpec) Descriptor() ([]byte, []int) {
	return fileDescriptor_4c16552f9fdb66d8, []int{0}
}

func (m *ModelSpec) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ModelSpec.Unmarshal(m, b)
}
func (m *ModelSpec) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ModelSpec.Marshal(b, m, deterministic)
}
func (m *ModelSpec) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ModelSpec.Merge(m, src)
}
func (m *ModelSpec) XXX_Size() int {
	return xxx_messageInfo_ModelSpec.Size(m)
}
func (m *ModelSpec) XXX_DiscardUnknown() {
	xxx_messageInfo_ModelSpec.DiscardUnknown(m)
}

var xxx_messageInfo_ModelSpec proto.InternalMessageInfo

func (m *ModelSpec) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *ModelSpec) GetVersion() *wrappers.Int64Value {
	if m != nil {
		return m.Version
	}
	return nil
}

func (m *ModelSpec) GetSignatureName() string {
	if m != nil {
		return m.SignatureName
	}
	return ""
}

func init() {
	proto.RegisterType((*ModelSpec)(nil), "tensorflow.serving.ModelSpec")
}

func init() { proto.RegisterFile("model.proto", fileDescriptor_4c16552f9fdb66d8) }

var fileDescriptor_4c16552f9fdb66d8 = []byte{
	// 186 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xe2, 0xce, 0xcd, 0x4f, 0x49,
	0xcd, 0xd1, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0x12, 0x2a, 0x49, 0xcd, 0x2b, 0xce, 0x2f, 0x4a,
	0xcb, 0xc9, 0x2f, 0xd7, 0x2b, 0x4e, 0x2d, 0x2a, 0xcb, 0xcc, 0x4b, 0x97, 0x92, 0x4b, 0xcf, 0xcf,
	0x4f, 0xcf, 0x49, 0xd5, 0x07, 0xab, 0x48, 0x2a, 0x4d, 0xd3, 0x2f, 0x2f, 0x4a, 0x2c, 0x28, 0x48,
	0x2d, 0x2a, 0x86, 0xe8, 0x51, 0xaa, 0xe5, 0xe2, 0xf4, 0x05, 0x19, 0x11, 0x5c, 0x90, 0x9a, 0x2c,
	0x24, 0xc4, 0xc5, 0x92, 0x97, 0x98, 0x9b, 0x2a, 0xc1, 0xa8, 0xc0, 0xa8, 0xc1, 0x19, 0x04, 0x66,
	0x0b, 0x99, 0x72, 0xb1, 0x97, 0xa5, 0x16, 0x15, 0x67, 0xe6, 0xe7, 0x49, 0x30, 0x29, 0x30, 0x6a,
	0x70, 0x1b, 0x49, 0xeb, 0x41, 0x8c, 0xd4, 0x83, 0x19, 0xa9, 0xe7, 0x99, 0x57, 0x62, 0x66, 0x12,
	0x96, 0x98, 0x53, 0x9a, 0x1a, 0x04, 0x53, 0x2b, 0xa4, 0xca, 0xc5, 0x57, 0x9c, 0x99, 0x9e, 0x97,
	0x58, 0x52, 0x5a, 0x94, 0x1a, 0x0f, 0x36, 0x94, 0x19, 0x6c, 0x28, 0x2f, 0x5c, 0xd4, 0x2f, 0x31,
	0x37, 0xd5, 0x89, 0xf9, 0x07, 0x23, 0x63, 0x12, 0x1b, 0xd8, 0x24, 0x63, 0x40, 0x00, 0x00, 0x00,
	0xff, 0xff, 0xe5, 0x39, 0x0b, 0x5f, 0xcd, 0x00, 0x00, 0x00,
}