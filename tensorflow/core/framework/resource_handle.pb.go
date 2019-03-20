// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/framework/resource_handle.proto

package tensorflow

import (
	fmt "fmt"
	math "math"

	proto "github.com/golang/protobuf/proto"
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

// Protocol buffer representing a handle to a tensorflow resource. Handles are
// not valid across executions, but can be serialized back and forth from within
// a single run.
type ResourceHandle struct {
	// Unique name for the device containing the resource.
	Device string `protobuf:"bytes,1,opt,name=device,proto3" json:"device,omitempty"`
	// Container in which this resource is placed.
	Container string `protobuf:"bytes,2,opt,name=container,proto3" json:"container,omitempty"`
	// Unique name of this resource.
	Name string `protobuf:"bytes,3,opt,name=name,proto3" json:"name,omitempty"`
	// Hash code for the type of the resource. Is only valid in the same device
	// and in the same execution.
	HashCode uint64 `protobuf:"varint,4,opt,name=hash_code,json=hashCode,proto3" json:"hash_code,omitempty"`
	// For debug-only, the name of the type pointed to by this handle, if
	// available.
	MaybeTypeName        string   `protobuf:"bytes,5,opt,name=maybe_type_name,json=maybeTypeName,proto3" json:"maybe_type_name,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ResourceHandle) Reset()         { *m = ResourceHandle{} }
func (m *ResourceHandle) String() string { return proto.CompactTextString(m) }
func (*ResourceHandle) ProtoMessage()    {}
func (*ResourceHandle) Descriptor() ([]byte, []int) {
	return fileDescriptor_a36024d2bd9a2afd, []int{0}
}

func (m *ResourceHandle) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ResourceHandle.Unmarshal(m, b)
}
func (m *ResourceHandle) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ResourceHandle.Marshal(b, m, deterministic)
}
func (m *ResourceHandle) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ResourceHandle.Merge(m, src)
}
func (m *ResourceHandle) XXX_Size() int {
	return xxx_messageInfo_ResourceHandle.Size(m)
}
func (m *ResourceHandle) XXX_DiscardUnknown() {
	xxx_messageInfo_ResourceHandle.DiscardUnknown(m)
}

var xxx_messageInfo_ResourceHandle proto.InternalMessageInfo

func (m *ResourceHandle) GetDevice() string {
	if m != nil {
		return m.Device
	}
	return ""
}

func (m *ResourceHandle) GetContainer() string {
	if m != nil {
		return m.Container
	}
	return ""
}

func (m *ResourceHandle) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *ResourceHandle) GetHashCode() uint64 {
	if m != nil {
		return m.HashCode
	}
	return 0
}

func (m *ResourceHandle) GetMaybeTypeName() string {
	if m != nil {
		return m.MaybeTypeName
	}
	return ""
}

func init() {
	proto.RegisterType((*ResourceHandle)(nil), "tensorflow.ResourceHandle")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/resource_handle.proto", fileDescriptor_a36024d2bd9a2afd)
}

var fileDescriptor_a36024d2bd9a2afd = []byte{
	// 223 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x54, 0x8f, 0xcf, 0x4a, 0xc3, 0x40,
	0x10, 0xc6, 0x59, 0x8d, 0xc5, 0x0c, 0xa8, 0xb0, 0x82, 0x2c, 0xe8, 0xa1, 0x78, 0x90, 0x9e, 0x92,
	0x83, 0x3e, 0x41, 0xbd, 0x78, 0x92, 0x12, 0xbc, 0x87, 0xed, 0xe6, 0xab, 0x29, 0x36, 0x3b, 0x61,
	0xb2, 0x5a, 0xf2, 0x34, 0xbe, 0xa6, 0x47, 0xe9, 0x50, 0x0c, 0xde, 0x66, 0xbe, 0x3f, 0xf0, 0xfd,
	0xa8, 0x4c, 0x88, 0x03, 0xcb, 0x66, 0xc7, 0xfb, 0x32, 0xb0, 0xa0, 0xdc, 0x88, 0xef, 0xb0, 0x67,
	0xf9, 0x28, 0x05, 0x03, 0x7f, 0x4a, 0x40, 0xdd, 0xfa, 0xd8, 0xec, 0x50, 0xf4, 0xc2, 0x89, 0x2d,
	0x4d, 0x85, 0xfb, 0x6f, 0x43, 0x97, 0xd5, 0x31, 0xf5, 0xa2, 0x21, 0x7b, 0x43, 0xb3, 0x06, 0x5f,
	0xdb, 0x00, 0x67, 0xe6, 0x66, 0x91, 0x57, 0xc7, 0xcf, 0xde, 0x51, 0x1e, 0x38, 0x26, 0xbf, 0x8d,
	0x10, 0x77, 0xa2, 0xd6, 0x24, 0x58, 0x4b, 0x59, 0xf4, 0x1d, 0xdc, 0xa9, 0x1a, 0x7a, 0xdb, 0x5b,
	0xca, 0x5b, 0x3f, 0xb4, 0x75, 0xe0, 0x06, 0x2e, 0x9b, 0x9b, 0x45, 0x56, 0x9d, 0x1f, 0x84, 0x67,
	0x6e, 0x60, 0x1f, 0xe8, 0xaa, 0xf3, 0xe3, 0x1a, 0x75, 0x1a, 0x7b, 0xd4, 0xda, 0x3d, 0xd3, 0xee,
	0x85, 0xca, 0x6f, 0x63, 0x8f, 0x57, 0xdf, 0x61, 0xf9, 0x44, 0x8e, 0xe5, 0xbd, 0x98, 0x36, 0x17,
	0x7f, 0x7c, 0xcb, 0xeb, 0xff, 0xd3, 0x57, 0x07, 0xbc, 0x95, 0xf9, 0x31, 0x66, 0x3d, 0x53, 0xd4,
	0xc7, 0xdf, 0x00, 0x00, 0x00, 0xff, 0xff, 0x39, 0x9f, 0x89, 0xe8, 0x1d, 0x01, 0x00, 0x00,
}
