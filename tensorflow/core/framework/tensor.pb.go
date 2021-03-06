// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/framework/tensor.proto

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

// Protocol buffer representing a tensor.
type TensorProto struct {
	Dtype DataType `protobuf:"varint,1,opt,name=dtype,proto3,enum=tensorflow.DataType" json:"dtype,omitempty"`
	// Shape of the tensor.  TODO(touts): sort out the 0-rank issues.
	TensorShape *TensorShapeProto `protobuf:"bytes,2,opt,name=tensor_shape,json=tensorShape,proto3" json:"tensor_shape,omitempty"`
	// Version number.
	//
	// In version 0, if the "repeated xxx" representations contain only one
	// element, that element is repeated to fill the shape.  This makes it easy
	// to represent a constant Tensor with a single value.
	VersionNumber int32 `protobuf:"varint,3,opt,name=version_number,json=versionNumber,proto3" json:"version_number,omitempty"`
	// Serialized content from Tensor::AsProtoTensorContent(). This representation
	// can be used for all tensor types.
	TensorContent []byte `protobuf:"bytes,4,opt,name=tensor_content,json=tensorContent,proto3" json:"tensor_content,omitempty"`
	// DT_HALF. Note that since protobuf has no int16 type, we'll have some
	// pointless zero padding for each value here.
	HalfVal []int32 `protobuf:"varint,13,rep,packed,name=half_val,json=halfVal,proto3" json:"half_val,omitempty"`
	// DT_FLOAT.
	FloatVal []float32 `protobuf:"fixed32,5,rep,packed,name=float_val,json=floatVal,proto3" json:"float_val,omitempty"`
	// DT_DOUBLE.
	DoubleVal []float64 `protobuf:"fixed64,6,rep,packed,name=double_val,json=doubleVal,proto3" json:"double_val,omitempty"`
	// DT_INT32, DT_INT16, DT_INT8, DT_UINT8.
	IntVal []int32 `protobuf:"varint,7,rep,packed,name=int_val,json=intVal,proto3" json:"int_val,omitempty"`
	// DT_STRING
	StringVal [][]byte `protobuf:"bytes,8,rep,name=string_val,json=stringVal,proto3" json:"string_val,omitempty"`
	// DT_COMPLEX64. scomplex_val(2*i) and scomplex_val(2*i+1) are real
	// and imaginary parts of i-th single precision complex.
	ScomplexVal []float32 `protobuf:"fixed32,9,rep,packed,name=scomplex_val,json=scomplexVal,proto3" json:"scomplex_val,omitempty"`
	// DT_INT64
	Int64Val []int64 `protobuf:"varint,10,rep,packed,name=int64_val,json=int64Val,proto3" json:"int64_val,omitempty"`
	// DT_BOOL
	BoolVal []bool `protobuf:"varint,11,rep,packed,name=bool_val,json=boolVal,proto3" json:"bool_val,omitempty"`
	// DT_COMPLEX128. dcomplex_val(2*i) and dcomplex_val(2*i+1) are real
	// and imaginary parts of i-th double precision complex.
	DcomplexVal []float64 `protobuf:"fixed64,12,rep,packed,name=dcomplex_val,json=dcomplexVal,proto3" json:"dcomplex_val,omitempty"`
	// DT_RESOURCE
	ResourceHandleVal    []*ResourceHandle `protobuf:"bytes,14,rep,name=resource_handle_val,json=resourceHandleVal,proto3" json:"resource_handle_val,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *TensorProto) Reset()         { *m = TensorProto{} }
func (m *TensorProto) String() string { return proto.CompactTextString(m) }
func (*TensorProto) ProtoMessage()    {}
func (*TensorProto) Descriptor() ([]byte, []int) {
	return fileDescriptor_efa68180bc31e4fc, []int{0}
}

func (m *TensorProto) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_TensorProto.Unmarshal(m, b)
}
func (m *TensorProto) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_TensorProto.Marshal(b, m, deterministic)
}
func (m *TensorProto) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TensorProto.Merge(m, src)
}
func (m *TensorProto) XXX_Size() int {
	return xxx_messageInfo_TensorProto.Size(m)
}
func (m *TensorProto) XXX_DiscardUnknown() {
	xxx_messageInfo_TensorProto.DiscardUnknown(m)
}

var xxx_messageInfo_TensorProto proto.InternalMessageInfo

func (m *TensorProto) GetDtype() DataType {
	if m != nil {
		return m.Dtype
	}
	return DataType_DT_INVALID
}

func (m *TensorProto) GetTensorShape() *TensorShapeProto {
	if m != nil {
		return m.TensorShape
	}
	return nil
}

func (m *TensorProto) GetVersionNumber() int32 {
	if m != nil {
		return m.VersionNumber
	}
	return 0
}

func (m *TensorProto) GetTensorContent() []byte {
	if m != nil {
		return m.TensorContent
	}
	return nil
}

func (m *TensorProto) GetHalfVal() []int32 {
	if m != nil {
		return m.HalfVal
	}
	return nil
}

func (m *TensorProto) GetFloatVal() []float32 {
	if m != nil {
		return m.FloatVal
	}
	return nil
}

func (m *TensorProto) GetDoubleVal() []float64 {
	if m != nil {
		return m.DoubleVal
	}
	return nil
}

func (m *TensorProto) GetIntVal() []int32 {
	if m != nil {
		return m.IntVal
	}
	return nil
}

func (m *TensorProto) GetStringVal() [][]byte {
	if m != nil {
		return m.StringVal
	}
	return nil
}

func (m *TensorProto) GetScomplexVal() []float32 {
	if m != nil {
		return m.ScomplexVal
	}
	return nil
}

func (m *TensorProto) GetInt64Val() []int64 {
	if m != nil {
		return m.Int64Val
	}
	return nil
}

func (m *TensorProto) GetBoolVal() []bool {
	if m != nil {
		return m.BoolVal
	}
	return nil
}

func (m *TensorProto) GetDcomplexVal() []float64 {
	if m != nil {
		return m.DcomplexVal
	}
	return nil
}

func (m *TensorProto) GetResourceHandleVal() []*ResourceHandle {
	if m != nil {
		return m.ResourceHandleVal
	}
	return nil
}

func init() {
	proto.RegisterType((*TensorProto)(nil), "tensorflow.TensorProto")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/tensor.proto", fileDescriptor_efa68180bc31e4fc)
}

var fileDescriptor_efa68180bc31e4fc = []byte{
	// 431 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x92, 0x41, 0x8f, 0xd3, 0x30,
	0x10, 0x85, 0xe5, 0x7a, 0xdb, 0xa6, 0x93, 0xb4, 0x82, 0xc0, 0x21, 0x2a, 0xac, 0x30, 0x48, 0x45,
	0x16, 0x82, 0x56, 0x2a, 0x88, 0x2b, 0x52, 0xe1, 0x80, 0x38, 0xa0, 0x55, 0x58, 0xed, 0xb5, 0x72,
	0x1b, 0x77, 0x1b, 0xe1, 0xda, 0x91, 0xe3, 0xee, 0xb2, 0x3f, 0x8f, 0x7f, 0xc5, 0x11, 0x79, 0x9c,
	0xb6, 0x01, 0x09, 0xf6, 0x98, 0xf7, 0xbe, 0x37, 0x6f, 0xac, 0x0c, 0xbc, 0x74, 0x52, 0xd7, 0xc6,
	0x6e, 0x94, 0xb9, 0x9d, 0xad, 0x8d, 0x95, 0xb3, 0x8d, 0x15, 0x3b, 0x79, 0x6b, 0xec, 0xf7, 0x59,
	0x70, 0xa6, 0x95, 0x35, 0xce, 0xa4, 0x70, 0xe2, 0xc6, 0xb3, 0x7f, 0x67, 0xac, 0xac, 0xcd, 0xde,
	0xae, 0xe5, 0x72, 0x2b, 0x74, 0xa1, 0x64, 0x08, 0x8f, 0x5f, 0xdf, 0x57, 0xb2, 0xac, 0xb7, 0xa2,
	0x3a, 0xd0, 0x93, 0xff, 0xd0, 0x77, 0x95, 0xac, 0x03, 0xf6, 0xe2, 0xe7, 0x19, 0xc4, 0x97, 0x48,
	0x5e, 0xe0, 0x86, 0xaf, 0xa0, 0x5b, 0x78, 0x3f, 0x23, 0x8c, 0xf0, 0xd1, 0xfc, 0xf1, 0xf4, 0x34,
	0x66, 0xfa, 0x49, 0x38, 0x71, 0x79, 0x57, 0xc9, 0x3c, 0x20, 0xe9, 0x07, 0x48, 0xda, 0xc5, 0x59,
	0x87, 0x11, 0x1e, 0xcf, 0x9f, 0xb6, 0x23, 0x61, 0xf4, 0x37, 0x6f, 0xe3, 0xfc, 0x3c, 0x76, 0x27,
	0x25, 0x9d, 0xc0, 0xe8, 0x46, 0xda, 0xba, 0x34, 0x7a, 0xa9, 0xf7, 0xbb, 0x95, 0xb4, 0x19, 0x65,
	0x84, 0x77, 0xf3, 0x61, 0xa3, 0x7e, 0x45, 0xd1, 0x63, 0x4d, 0xcf, 0xda, 0x68, 0x27, 0xb5, 0xcb,
	0xce, 0x18, 0xe1, 0x49, 0x3e, 0x0c, 0xea, 0xc7, 0x20, 0xa6, 0xe7, 0x10, 0x6d, 0x85, 0xda, 0x2c,
	0x6f, 0x84, 0xca, 0x86, 0x8c, 0xf2, 0xee, 0xa2, 0xf3, 0x80, 0xe4, 0x7d, 0xaf, 0x5d, 0x09, 0x95,
	0x3e, 0x83, 0xc1, 0x46, 0x19, 0xe1, 0xd0, 0xef, 0x32, 0xca, 0x3b, 0xe8, 0x47, 0x28, 0x7a, 0xe0,
	0x39, 0x40, 0x61, 0xf6, 0x2b, 0x25, 0x91, 0xe8, 0x31, 0xca, 0x09, 0x12, 0x83, 0xa0, 0x7a, 0xe4,
	0x09, 0xf4, 0x4b, 0x1d, 0x26, 0xf4, 0x8f, 0x0d, 0xbd, 0x52, 0x63, 0xfe, 0x1c, 0xa0, 0x76, 0xb6,
	0xd4, 0xd7, 0xe8, 0x47, 0x8c, 0xf2, 0x24, 0x1f, 0x04, 0xc5, 0xdb, 0x13, 0x48, 0xea, 0xb5, 0xd9,
	0x55, 0x4a, 0xfe, 0x40, 0x60, 0x70, 0x5c, 0x21, 0x3e, 0xe8, 0xcd, 0x9a, 0xa5, 0x76, 0xef, 0xdf,
	0x21, 0x03, 0x8c, 0x72, 0x1a, 0xd6, 0x44, 0x31, 0xd4, 0x44, 0x2b, 0x63, 0x14, 0xfa, 0x31, 0xa3,
	0x3c, 0x0a, 0xcf, 0xf4, 0x5a, 0x53, 0x53, 0xb4, 0x6b, 0x92, 0xe3, 0x3b, 0xe2, 0xa2, 0x55, 0xf3,
	0x05, 0x1e, 0xfd, 0x75, 0x65, 0x48, 0x8f, 0x18, 0xe5, 0xf1, 0x7c, 0xdc, 0xfe, 0x85, 0x79, 0x83,
	0x7d, 0x46, 0x2a, 0x7f, 0x68, 0xff, 0xf8, 0xbe, 0x12, 0x6a, 0xf1, 0x06, 0x32, 0x63, 0xaf, 0xdb,
	0x99, 0xe3, 0xad, 0x2d, 0x92, 0xd6, 0x71, 0xd5, 0x17, 0xe4, 0x17, 0x21, 0xab, 0x1e, 0x5e, 0xde,
	0xdb, 0xdf, 0x01, 0x00, 0x00, 0xff, 0xff, 0xec, 0x7c, 0x59, 0xcf, 0x35, 0x03, 0x00, 0x00,
}
