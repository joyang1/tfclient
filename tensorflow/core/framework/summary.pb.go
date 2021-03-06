// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/framework/summary.proto

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

// Metadata associated with a series of Summary data
type SummaryDescription struct {
	// Hint on how plugins should process the data in this series.
	// Supported values include "scalar", "histogram", "image", "audio"
	TypeHint             string   `protobuf:"bytes,1,opt,name=type_hint,json=typeHint,proto3" json:"type_hint,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *SummaryDescription) Reset()         { *m = SummaryDescription{} }
func (m *SummaryDescription) String() string { return proto.CompactTextString(m) }
func (*SummaryDescription) ProtoMessage()    {}
func (*SummaryDescription) Descriptor() ([]byte, []int) {
	return fileDescriptor_80d4b41d3e8d8b09, []int{0}
}

func (m *SummaryDescription) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_SummaryDescription.Unmarshal(m, b)
}
func (m *SummaryDescription) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_SummaryDescription.Marshal(b, m, deterministic)
}
func (m *SummaryDescription) XXX_Merge(src proto.Message) {
	xxx_messageInfo_SummaryDescription.Merge(m, src)
}
func (m *SummaryDescription) XXX_Size() int {
	return xxx_messageInfo_SummaryDescription.Size(m)
}
func (m *SummaryDescription) XXX_DiscardUnknown() {
	xxx_messageInfo_SummaryDescription.DiscardUnknown(m)
}

var xxx_messageInfo_SummaryDescription proto.InternalMessageInfo

func (m *SummaryDescription) GetTypeHint() string {
	if m != nil {
		return m.TypeHint
	}
	return ""
}

// Serialization format for histogram module in
// core/lib/histogram/histogram.h
type HistogramProto struct {
	Min        float64 `protobuf:"fixed64,1,opt,name=min,proto3" json:"min,omitempty"`
	Max        float64 `protobuf:"fixed64,2,opt,name=max,proto3" json:"max,omitempty"`
	Num        float64 `protobuf:"fixed64,3,opt,name=num,proto3" json:"num,omitempty"`
	Sum        float64 `protobuf:"fixed64,4,opt,name=sum,proto3" json:"sum,omitempty"`
	SumSquares float64 `protobuf:"fixed64,5,opt,name=sum_squares,json=sumSquares,proto3" json:"sum_squares,omitempty"`
	// Parallel arrays encoding the bucket boundaries and the bucket values.
	// bucket(i) is the count for the bucket i.  The range for
	// a bucket is:
	//   i == 0:  -DBL_MAX .. bucket_limit(0)
	//   i != 0:  bucket_limit(i-1) .. bucket_limit(i)
	BucketLimit          []float64 `protobuf:"fixed64,6,rep,packed,name=bucket_limit,json=bucketLimit,proto3" json:"bucket_limit,omitempty"`
	Bucket               []float64 `protobuf:"fixed64,7,rep,packed,name=bucket,proto3" json:"bucket,omitempty"`
	XXX_NoUnkeyedLiteral struct{}  `json:"-"`
	XXX_unrecognized     []byte    `json:"-"`
	XXX_sizecache        int32     `json:"-"`
}

func (m *HistogramProto) Reset()         { *m = HistogramProto{} }
func (m *HistogramProto) String() string { return proto.CompactTextString(m) }
func (*HistogramProto) ProtoMessage()    {}
func (*HistogramProto) Descriptor() ([]byte, []int) {
	return fileDescriptor_80d4b41d3e8d8b09, []int{1}
}

func (m *HistogramProto) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_HistogramProto.Unmarshal(m, b)
}
func (m *HistogramProto) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_HistogramProto.Marshal(b, m, deterministic)
}
func (m *HistogramProto) XXX_Merge(src proto.Message) {
	xxx_messageInfo_HistogramProto.Merge(m, src)
}
func (m *HistogramProto) XXX_Size() int {
	return xxx_messageInfo_HistogramProto.Size(m)
}
func (m *HistogramProto) XXX_DiscardUnknown() {
	xxx_messageInfo_HistogramProto.DiscardUnknown(m)
}

var xxx_messageInfo_HistogramProto proto.InternalMessageInfo

func (m *HistogramProto) GetMin() float64 {
	if m != nil {
		return m.Min
	}
	return 0
}

func (m *HistogramProto) GetMax() float64 {
	if m != nil {
		return m.Max
	}
	return 0
}

func (m *HistogramProto) GetNum() float64 {
	if m != nil {
		return m.Num
	}
	return 0
}

func (m *HistogramProto) GetSum() float64 {
	if m != nil {
		return m.Sum
	}
	return 0
}

func (m *HistogramProto) GetSumSquares() float64 {
	if m != nil {
		return m.SumSquares
	}
	return 0
}

func (m *HistogramProto) GetBucketLimit() []float64 {
	if m != nil {
		return m.BucketLimit
	}
	return nil
}

func (m *HistogramProto) GetBucket() []float64 {
	if m != nil {
		return m.Bucket
	}
	return nil
}

// A Summary is a set of named values to be displayed by the
// visualizer.
//
// Summaries are produced regularly during training, as controlled by
// the "summary_interval_secs" attribute of the training operation.
// Summaries are also produced at the end of an evaluation.
type Summary struct {
	// Set of values for the summary.
	Value                []*Summary_Value `protobuf:"bytes,1,rep,name=value,proto3" json:"value,omitempty"`
	XXX_NoUnkeyedLiteral struct{}         `json:"-"`
	XXX_unrecognized     []byte           `json:"-"`
	XXX_sizecache        int32            `json:"-"`
}

func (m *Summary) Reset()         { *m = Summary{} }
func (m *Summary) String() string { return proto.CompactTextString(m) }
func (*Summary) ProtoMessage()    {}
func (*Summary) Descriptor() ([]byte, []int) {
	return fileDescriptor_80d4b41d3e8d8b09, []int{2}
}

func (m *Summary) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Summary.Unmarshal(m, b)
}
func (m *Summary) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Summary.Marshal(b, m, deterministic)
}
func (m *Summary) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Summary.Merge(m, src)
}
func (m *Summary) XXX_Size() int {
	return xxx_messageInfo_Summary.Size(m)
}
func (m *Summary) XXX_DiscardUnknown() {
	xxx_messageInfo_Summary.DiscardUnknown(m)
}

var xxx_messageInfo_Summary proto.InternalMessageInfo

func (m *Summary) GetValue() []*Summary_Value {
	if m != nil {
		return m.Value
	}
	return nil
}

type Summary_Image struct {
	// Dimensions of the image.
	Height int32 `protobuf:"varint,1,opt,name=height,proto3" json:"height,omitempty"`
	Width  int32 `protobuf:"varint,2,opt,name=width,proto3" json:"width,omitempty"`
	// Valid colorspace values are
	//   1 - grayscale
	//   2 - grayscale + alpha
	//   3 - RGB
	//   4 - RGBA
	//   5 - DIGITAL_YUV
	//   6 - BGRA
	Colorspace int32 `protobuf:"varint,3,opt,name=colorspace,proto3" json:"colorspace,omitempty"`
	// Image data in encoded format.  All image formats supported by
	// image_codec::CoderUtil can be stored here.
	EncodedImageString   []byte   `protobuf:"bytes,4,opt,name=encoded_image_string,json=encodedImageString,proto3" json:"encoded_image_string,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Summary_Image) Reset()         { *m = Summary_Image{} }
func (m *Summary_Image) String() string { return proto.CompactTextString(m) }
func (*Summary_Image) ProtoMessage()    {}
func (*Summary_Image) Descriptor() ([]byte, []int) {
	return fileDescriptor_80d4b41d3e8d8b09, []int{2, 0}
}

func (m *Summary_Image) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Summary_Image.Unmarshal(m, b)
}
func (m *Summary_Image) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Summary_Image.Marshal(b, m, deterministic)
}
func (m *Summary_Image) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Summary_Image.Merge(m, src)
}
func (m *Summary_Image) XXX_Size() int {
	return xxx_messageInfo_Summary_Image.Size(m)
}
func (m *Summary_Image) XXX_DiscardUnknown() {
	xxx_messageInfo_Summary_Image.DiscardUnknown(m)
}

var xxx_messageInfo_Summary_Image proto.InternalMessageInfo

func (m *Summary_Image) GetHeight() int32 {
	if m != nil {
		return m.Height
	}
	return 0
}

func (m *Summary_Image) GetWidth() int32 {
	if m != nil {
		return m.Width
	}
	return 0
}

func (m *Summary_Image) GetColorspace() int32 {
	if m != nil {
		return m.Colorspace
	}
	return 0
}

func (m *Summary_Image) GetEncodedImageString() []byte {
	if m != nil {
		return m.EncodedImageString
	}
	return nil
}

type Summary_Audio struct {
	// Sample rate of the audio in Hz.
	SampleRate float32 `protobuf:"fixed32,1,opt,name=sample_rate,json=sampleRate,proto3" json:"sample_rate,omitempty"`
	// Number of channels of audio.
	NumChannels int64 `protobuf:"varint,2,opt,name=num_channels,json=numChannels,proto3" json:"num_channels,omitempty"`
	// Length of the audio in frames (samples per channel).
	LengthFrames int64 `protobuf:"varint,3,opt,name=length_frames,json=lengthFrames,proto3" json:"length_frames,omitempty"`
	// Encoded audio data and its associated RFC 2045 content type (e.g.
	// "audio/wav").
	EncodedAudioString   []byte   `protobuf:"bytes,4,opt,name=encoded_audio_string,json=encodedAudioString,proto3" json:"encoded_audio_string,omitempty"`
	ContentType          string   `protobuf:"bytes,5,opt,name=content_type,json=contentType,proto3" json:"content_type,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Summary_Audio) Reset()         { *m = Summary_Audio{} }
func (m *Summary_Audio) String() string { return proto.CompactTextString(m) }
func (*Summary_Audio) ProtoMessage()    {}
func (*Summary_Audio) Descriptor() ([]byte, []int) {
	return fileDescriptor_80d4b41d3e8d8b09, []int{2, 1}
}

func (m *Summary_Audio) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Summary_Audio.Unmarshal(m, b)
}
func (m *Summary_Audio) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Summary_Audio.Marshal(b, m, deterministic)
}
func (m *Summary_Audio) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Summary_Audio.Merge(m, src)
}
func (m *Summary_Audio) XXX_Size() int {
	return xxx_messageInfo_Summary_Audio.Size(m)
}
func (m *Summary_Audio) XXX_DiscardUnknown() {
	xxx_messageInfo_Summary_Audio.DiscardUnknown(m)
}

var xxx_messageInfo_Summary_Audio proto.InternalMessageInfo

func (m *Summary_Audio) GetSampleRate() float32 {
	if m != nil {
		return m.SampleRate
	}
	return 0
}

func (m *Summary_Audio) GetNumChannels() int64 {
	if m != nil {
		return m.NumChannels
	}
	return 0
}

func (m *Summary_Audio) GetLengthFrames() int64 {
	if m != nil {
		return m.LengthFrames
	}
	return 0
}

func (m *Summary_Audio) GetEncodedAudioString() []byte {
	if m != nil {
		return m.EncodedAudioString
	}
	return nil
}

func (m *Summary_Audio) GetContentType() string {
	if m != nil {
		return m.ContentType
	}
	return ""
}

type Summary_Value struct {
	// Name of the node that output this summary; in general, the name of a
	// TensorSummary node. If the node in question has multiple outputs, then
	// a ":\d+" suffix will be appended, like "some_op:13".
	// Might not be set for legacy summaries (i.e. those not using the tensor
	// value field)
	NodeName string `protobuf:"bytes,7,opt,name=node_name,json=nodeName,proto3" json:"node_name,omitempty"`
	// Tag name for the data.  Will only be used by legacy summaries
	// (ie. those not using the tensor value field)
	// For legacy summaries, will be used as the title of the graph
	// in the visualizer.
	//
	// Tag is usually "op_name:value_name", where "op_name" itself can have
	// structure to indicate grouping.
	Tag string `protobuf:"bytes,1,opt,name=tag,proto3" json:"tag,omitempty"`
	// Value associated with the tag.
	//
	// Types that are valid to be assigned to Value:
	//	*Summary_Value_SimpleValue
	//	*Summary_Value_ObsoleteOldStyleHistogram
	//	*Summary_Value_Image
	//	*Summary_Value_Histo
	//	*Summary_Value_Audio
	//	*Summary_Value_Tensor
	Value                isSummary_Value_Value `protobuf_oneof:"value"`
	XXX_NoUnkeyedLiteral struct{}              `json:"-"`
	XXX_unrecognized     []byte                `json:"-"`
	XXX_sizecache        int32                 `json:"-"`
}

func (m *Summary_Value) Reset()         { *m = Summary_Value{} }
func (m *Summary_Value) String() string { return proto.CompactTextString(m) }
func (*Summary_Value) ProtoMessage()    {}
func (*Summary_Value) Descriptor() ([]byte, []int) {
	return fileDescriptor_80d4b41d3e8d8b09, []int{2, 2}
}

func (m *Summary_Value) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Summary_Value.Unmarshal(m, b)
}
func (m *Summary_Value) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Summary_Value.Marshal(b, m, deterministic)
}
func (m *Summary_Value) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Summary_Value.Merge(m, src)
}
func (m *Summary_Value) XXX_Size() int {
	return xxx_messageInfo_Summary_Value.Size(m)
}
func (m *Summary_Value) XXX_DiscardUnknown() {
	xxx_messageInfo_Summary_Value.DiscardUnknown(m)
}

var xxx_messageInfo_Summary_Value proto.InternalMessageInfo

func (m *Summary_Value) GetNodeName() string {
	if m != nil {
		return m.NodeName
	}
	return ""
}

func (m *Summary_Value) GetTag() string {
	if m != nil {
		return m.Tag
	}
	return ""
}

type isSummary_Value_Value interface {
	isSummary_Value_Value()
}

type Summary_Value_SimpleValue struct {
	SimpleValue float32 `protobuf:"fixed32,2,opt,name=simple_value,json=simpleValue,proto3,oneof"`
}

type Summary_Value_ObsoleteOldStyleHistogram struct {
	ObsoleteOldStyleHistogram []byte `protobuf:"bytes,3,opt,name=obsolete_old_style_histogram,json=obsoleteOldStyleHistogram,proto3,oneof"`
}

type Summary_Value_Image struct {
	Image *Summary_Image `protobuf:"bytes,4,opt,name=image,proto3,oneof"`
}

type Summary_Value_Histo struct {
	Histo *HistogramProto `protobuf:"bytes,5,opt,name=histo,proto3,oneof"`
}

type Summary_Value_Audio struct {
	Audio *Summary_Audio `protobuf:"bytes,6,opt,name=audio,proto3,oneof"`
}

type Summary_Value_Tensor struct {
	Tensor *TensorProto `protobuf:"bytes,8,opt,name=tensor,proto3,oneof"`
}

func (*Summary_Value_SimpleValue) isSummary_Value_Value() {}

func (*Summary_Value_ObsoleteOldStyleHistogram) isSummary_Value_Value() {}

func (*Summary_Value_Image) isSummary_Value_Value() {}

func (*Summary_Value_Histo) isSummary_Value_Value() {}

func (*Summary_Value_Audio) isSummary_Value_Value() {}

func (*Summary_Value_Tensor) isSummary_Value_Value() {}

func (m *Summary_Value) GetValue() isSummary_Value_Value {
	if m != nil {
		return m.Value
	}
	return nil
}

func (m *Summary_Value) GetSimpleValue() float32 {
	if x, ok := m.GetValue().(*Summary_Value_SimpleValue); ok {
		return x.SimpleValue
	}
	return 0
}

func (m *Summary_Value) GetObsoleteOldStyleHistogram() []byte {
	if x, ok := m.GetValue().(*Summary_Value_ObsoleteOldStyleHistogram); ok {
		return x.ObsoleteOldStyleHistogram
	}
	return nil
}

func (m *Summary_Value) GetImage() *Summary_Image {
	if x, ok := m.GetValue().(*Summary_Value_Image); ok {
		return x.Image
	}
	return nil
}

func (m *Summary_Value) GetHisto() *HistogramProto {
	if x, ok := m.GetValue().(*Summary_Value_Histo); ok {
		return x.Histo
	}
	return nil
}

func (m *Summary_Value) GetAudio() *Summary_Audio {
	if x, ok := m.GetValue().(*Summary_Value_Audio); ok {
		return x.Audio
	}
	return nil
}

func (m *Summary_Value) GetTensor() *TensorProto {
	if x, ok := m.GetValue().(*Summary_Value_Tensor); ok {
		return x.Tensor
	}
	return nil
}

// XXX_OneofWrappers is for the internal use of the proto package.
func (*Summary_Value) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*Summary_Value_SimpleValue)(nil),
		(*Summary_Value_ObsoleteOldStyleHistogram)(nil),
		(*Summary_Value_Image)(nil),
		(*Summary_Value_Histo)(nil),
		(*Summary_Value_Audio)(nil),
		(*Summary_Value_Tensor)(nil),
	}
}

func init() {
	proto.RegisterType((*SummaryDescription)(nil), "tensorflow.SummaryDescription")
	proto.RegisterType((*HistogramProto)(nil), "tensorflow.HistogramProto")
	proto.RegisterType((*Summary)(nil), "tensorflow.Summary")
	proto.RegisterType((*Summary_Image)(nil), "tensorflow.Summary.Image")
	proto.RegisterType((*Summary_Audio)(nil), "tensorflow.Summary.Audio")
	proto.RegisterType((*Summary_Value)(nil), "tensorflow.Summary.Value")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/summary.proto", fileDescriptor_80d4b41d3e8d8b09)
}

var fileDescriptor_80d4b41d3e8d8b09 = []byte{
	// 632 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x94, 0xcf, 0x6e, 0x13, 0x3b,
	0x14, 0xc6, 0x3b, 0xc9, 0x9d, 0x49, 0x7b, 0x26, 0xbd, 0xaa, 0xac, 0xea, 0xde, 0x69, 0xee, 0x15,
	0x94, 0x56, 0x40, 0x57, 0x09, 0x2d, 0x4f, 0xd0, 0x80, 0x50, 0x90, 0x10, 0x54, 0x4e, 0xc5, 0x76,
	0xe4, 0xce, 0xb8, 0x13, 0xab, 0x63, 0x3b, 0x8c, 0x6d, 0xda, 0xac, 0x59, 0xf0, 0x52, 0x6c, 0x79,
	0x23, 0x16, 0x2c, 0x91, 0x8f, 0xdd, 0x36, 0x48, 0x74, 0x67, 0xff, 0xfc, 0x1d, 0x9f, 0x3f, 0xfe,
	0x66, 0xe0, 0xb9, 0xe5, 0xca, 0xe8, 0xee, 0xb2, 0xd5, 0xd7, 0x93, 0x4a, 0x77, 0x7c, 0x72, 0xd9,
	0x31, 0xc9, 0xaf, 0x75, 0x77, 0x35, 0x31, 0x4e, 0x4a, 0xd6, 0xad, 0xc6, 0xcb, 0x4e, 0x5b, 0x4d,
	0xe0, 0x5e, 0x38, 0x7a, 0xf6, 0x70, 0x50, 0x38, 0x09, 0x31, 0x07, 0xc7, 0x40, 0xe6, 0xe1, 0x92,
	0xd7, 0xdc, 0x54, 0x9d, 0x58, 0x5a, 0xa1, 0x15, 0xf9, 0x0f, 0xb6, 0xec, 0x6a, 0xc9, 0xcb, 0x85,
	0x50, 0xb6, 0x48, 0xf6, 0x93, 0xa3, 0x2d, 0xba, 0xe9, 0xc1, 0x4c, 0x28, 0x7b, 0xf0, 0x2d, 0x81,
	0xbf, 0x67, 0xc2, 0x58, 0xdd, 0x74, 0x4c, 0x9e, 0x61, 0xe6, 0x1d, 0xe8, 0x4b, 0xa1, 0x50, 0x99,
	0x50, 0xbf, 0x44, 0xc2, 0x6e, 0x8a, 0x5e, 0x24, 0xec, 0xc6, 0x13, 0xe5, 0x64, 0xd1, 0x0f, 0x44,
	0x39, 0xe9, 0x89, 0x71, 0xb2, 0xf8, 0x2b, 0x10, 0xe3, 0x24, 0x79, 0x0c, 0xb9, 0x71, 0xb2, 0x34,
	0x9f, 0x1c, 0xeb, 0xb8, 0x29, 0x52, 0x3c, 0x01, 0xe3, 0xe4, 0x3c, 0x10, 0xf2, 0x14, 0x86, 0x17,
	0xae, 0xba, 0xe2, 0xb6, 0x6c, 0x85, 0x14, 0xb6, 0xc8, 0xf6, 0xfb, 0x47, 0xc9, 0xb4, 0xb7, 0x93,
	0xd0, 0x3c, 0xf0, 0x77, 0x1e, 0x93, 0x11, 0x64, 0x61, 0x5b, 0x0c, 0xee, 0x04, 0x91, 0x1c, 0x7c,
	0xc9, 0x60, 0x10, 0x5b, 0x26, 0x13, 0x48, 0x3f, 0xb3, 0xd6, 0xf1, 0x22, 0xd9, 0xef, 0x1f, 0xe5,
	0x27, 0x7b, 0xe3, 0xfb, 0xa9, 0x8d, 0xa3, 0x66, 0xfc, 0xd1, 0x0b, 0x68, 0xd0, 0x8d, 0xbe, 0x26,
	0x90, 0xbe, 0x95, 0xac, 0xe1, 0xe4, 0x1f, 0xc8, 0x16, 0x5c, 0x34, 0x8b, 0x30, 0x9f, 0x94, 0xc6,
	0x1d, 0xd9, 0x85, 0xf4, 0x5a, 0xd4, 0x76, 0x81, 0xad, 0xa7, 0x34, 0x6c, 0xc8, 0x23, 0x80, 0x4a,
	0xb7, 0xba, 0x33, 0x4b, 0x56, 0x71, 0x9c, 0x41, 0x4a, 0xd7, 0x08, 0x79, 0x01, 0xbb, 0x5c, 0x55,
	0xba, 0xe6, 0x75, 0x29, 0xfc, 0xf5, 0xa5, 0xb1, 0x9d, 0x50, 0x0d, 0xce, 0x66, 0x48, 0x49, 0x3c,
	0xc3, 0xcc, 0x73, 0x3c, 0x19, 0x7d, 0x4f, 0x20, 0x3d, 0x75, 0xb5, 0xd0, 0x38, 0x34, 0x26, 0x97,
	0x2d, 0x2f, 0x3b, 0x66, 0x39, 0x96, 0xd3, 0xa3, 0x10, 0x10, 0x65, 0x96, 0x93, 0x27, 0x30, 0x54,
	0x4e, 0x96, 0xd5, 0x82, 0x29, 0xc5, 0x5b, 0x83, 0x95, 0xf5, 0x69, 0xae, 0x9c, 0x7c, 0x15, 0x11,
	0x39, 0x84, 0xed, 0x96, 0xab, 0xc6, 0x2e, 0x4a, 0xf4, 0x89, 0xc1, 0x12, 0xfb, 0x74, 0x18, 0xe0,
	0x1b, 0x64, 0xeb, 0x45, 0x32, 0x9f, 0xf9, 0xcf, 0x45, 0x62, 0x51, 0xa1, 0x48, 0x9f, 0xb9, 0xd2,
	0xca, 0x72, 0x65, 0x4b, 0x6f, 0x1f, 0x7c, 0xd0, 0x2d, 0x9a, 0x47, 0x76, 0xbe, 0x5a, 0xf2, 0xd1,
	0x8f, 0x1e, 0xa4, 0x38, 0x62, 0x6f, 0x3a, 0xa5, 0x6b, 0x5e, 0x2a, 0x26, 0x79, 0x31, 0x08, 0xa6,
	0xf3, 0xe0, 0x3d, 0x93, 0xdc, 0x7b, 0xc5, 0xb2, 0x26, 0x7a, 0xd1, 0x2f, 0xc9, 0x21, 0x0c, 0x8d,
	0xc0, 0xb6, 0xc3, 0x13, 0xfa, 0xae, 0x7a, 0xb3, 0x0d, 0x9a, 0x07, 0x1a, 0xee, 0x3c, 0x85, 0xff,
	0xf5, 0x85, 0xd1, 0x2d, 0xb7, 0xbc, 0xd4, 0x6d, 0x5d, 0x1a, 0xbb, 0x6a, 0xbd, 0xad, 0xa3, 0x7b,
	0xb1, 0xcd, 0xe1, 0x6c, 0x83, 0xee, 0xdd, 0xaa, 0x3e, 0xb4, 0xf5, 0xdc, 0x6b, 0xee, 0x0c, 0x4e,
	0x8e, 0x21, 0xc5, 0x27, 0xc1, 0x36, 0x1f, 0xf0, 0x08, 0x3e, 0xcc, 0x6c, 0x83, 0x06, 0x25, 0x39,
	0x81, 0x14, 0x53, 0x60, 0xbf, 0xf9, 0xc9, 0x68, 0x3d, 0xe4, 0xf7, 0x2f, 0xc7, 0xc7, 0xa0, 0xd4,
	0xa7, 0xc1, 0xa1, 0x16, 0xd9, 0xc3, 0x69, 0x70, 0xb4, 0x3e, 0x04, 0x95, 0xe4, 0x18, 0xb2, 0x20,
	0x2a, 0x36, 0x31, 0xe6, 0xdf, 0xf5, 0x98, 0x73, 0x5c, 0xde, 0x26, 0x89, 0xc2, 0xe9, 0x20, 0x1a,
	0x7e, 0x3a, 0x86, 0x42, 0x77, 0xcd, 0x7a, 0xc0, 0xdd, 0x0f, 0x62, 0xba, 0x1d, 0xf3, 0x61, 0xb0,
	0x39, 0x4b, 0x7e, 0x26, 0xc9, 0x45, 0x86, 0xbf, 0x8b, 0x97, 0xbf, 0x02, 0x00, 0x00, 0xff, 0xff,
	0x03, 0xa2, 0x20, 0x82, 0x8d, 0x04, 0x00, 0x00,
}
