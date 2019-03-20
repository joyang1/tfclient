// Code generated by protoc-gen-go. DO NOT EDIT.
// source: tensorflow/core/framework/step_stats.proto

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

type AllocatorMemoryUsed struct {
	AllocatorName        string   `protobuf:"bytes,1,opt,name=allocator_name,json=allocatorName,proto3" json:"allocator_name,omitempty"`
	TotalBytes           int64    `protobuf:"varint,2,opt,name=total_bytes,json=totalBytes,proto3" json:"total_bytes,omitempty"`
	PeakBytes            int64    `protobuf:"varint,3,opt,name=peak_bytes,json=peakBytes,proto3" json:"peak_bytes,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *AllocatorMemoryUsed) Reset()         { *m = AllocatorMemoryUsed{} }
func (m *AllocatorMemoryUsed) String() string { return proto.CompactTextString(m) }
func (*AllocatorMemoryUsed) ProtoMessage()    {}
func (*AllocatorMemoryUsed) Descriptor() ([]byte, []int) {
	return fileDescriptor_1e915309f7ed52e5, []int{0}
}

func (m *AllocatorMemoryUsed) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_AllocatorMemoryUsed.Unmarshal(m, b)
}
func (m *AllocatorMemoryUsed) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_AllocatorMemoryUsed.Marshal(b, m, deterministic)
}
func (m *AllocatorMemoryUsed) XXX_Merge(src proto.Message) {
	xxx_messageInfo_AllocatorMemoryUsed.Merge(m, src)
}
func (m *AllocatorMemoryUsed) XXX_Size() int {
	return xxx_messageInfo_AllocatorMemoryUsed.Size(m)
}
func (m *AllocatorMemoryUsed) XXX_DiscardUnknown() {
	xxx_messageInfo_AllocatorMemoryUsed.DiscardUnknown(m)
}

var xxx_messageInfo_AllocatorMemoryUsed proto.InternalMessageInfo

func (m *AllocatorMemoryUsed) GetAllocatorName() string {
	if m != nil {
		return m.AllocatorName
	}
	return ""
}

func (m *AllocatorMemoryUsed) GetTotalBytes() int64 {
	if m != nil {
		return m.TotalBytes
	}
	return 0
}

func (m *AllocatorMemoryUsed) GetPeakBytes() int64 {
	if m != nil {
		return m.PeakBytes
	}
	return 0
}

// Output sizes recorded for a single execution of a graph node.
type NodeOutput struct {
	Slot                 int32              `protobuf:"varint,1,opt,name=slot,proto3" json:"slot,omitempty"`
	TensorDescription    *TensorDescription `protobuf:"bytes,3,opt,name=tensor_description,json=tensorDescription,proto3" json:"tensor_description,omitempty"`
	XXX_NoUnkeyedLiteral struct{}           `json:"-"`
	XXX_unrecognized     []byte             `json:"-"`
	XXX_sizecache        int32              `json:"-"`
}

func (m *NodeOutput) Reset()         { *m = NodeOutput{} }
func (m *NodeOutput) String() string { return proto.CompactTextString(m) }
func (*NodeOutput) ProtoMessage()    {}
func (*NodeOutput) Descriptor() ([]byte, []int) {
	return fileDescriptor_1e915309f7ed52e5, []int{1}
}

func (m *NodeOutput) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_NodeOutput.Unmarshal(m, b)
}
func (m *NodeOutput) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_NodeOutput.Marshal(b, m, deterministic)
}
func (m *NodeOutput) XXX_Merge(src proto.Message) {
	xxx_messageInfo_NodeOutput.Merge(m, src)
}
func (m *NodeOutput) XXX_Size() int {
	return xxx_messageInfo_NodeOutput.Size(m)
}
func (m *NodeOutput) XXX_DiscardUnknown() {
	xxx_messageInfo_NodeOutput.DiscardUnknown(m)
}

var xxx_messageInfo_NodeOutput proto.InternalMessageInfo

func (m *NodeOutput) GetSlot() int32 {
	if m != nil {
		return m.Slot
	}
	return 0
}

func (m *NodeOutput) GetTensorDescription() *TensorDescription {
	if m != nil {
		return m.TensorDescription
	}
	return nil
}

// Time/size stats recorded for a single execution of a graph node.
type NodeExecStats struct {
	// TODO(tucker): Use some more compact form of node identity than
	// the full string name.  Either all processes should agree on a
	// global id (cost_id?) for each node, or we should use a hash of
	// the name.
	NodeName             string                   `protobuf:"bytes,1,opt,name=node_name,json=nodeName,proto3" json:"node_name,omitempty"`
	AllStartMicros       int64                    `protobuf:"varint,2,opt,name=all_start_micros,json=allStartMicros,proto3" json:"all_start_micros,omitempty"`
	OpStartRelMicros     int64                    `protobuf:"varint,3,opt,name=op_start_rel_micros,json=opStartRelMicros,proto3" json:"op_start_rel_micros,omitempty"`
	OpEndRelMicros       int64                    `protobuf:"varint,4,opt,name=op_end_rel_micros,json=opEndRelMicros,proto3" json:"op_end_rel_micros,omitempty"`
	AllEndRelMicros      int64                    `protobuf:"varint,5,opt,name=all_end_rel_micros,json=allEndRelMicros,proto3" json:"all_end_rel_micros,omitempty"`
	Memory               []*AllocatorMemoryUsed   `protobuf:"bytes,6,rep,name=memory,proto3" json:"memory,omitempty"`
	Output               []*NodeOutput            `protobuf:"bytes,7,rep,name=output,proto3" json:"output,omitempty"`
	TimelineLabel        string                   `protobuf:"bytes,8,opt,name=timeline_label,json=timelineLabel,proto3" json:"timeline_label,omitempty"`
	ScheduledMicros      int64                    `protobuf:"varint,9,opt,name=scheduled_micros,json=scheduledMicros,proto3" json:"scheduled_micros,omitempty"`
	ThreadId             uint32                   `protobuf:"varint,10,opt,name=thread_id,json=threadId,proto3" json:"thread_id,omitempty"`
	ReferencedTensor     []*AllocationDescription `protobuf:"bytes,11,rep,name=referenced_tensor,json=referencedTensor,proto3" json:"referenced_tensor,omitempty"`
	XXX_NoUnkeyedLiteral struct{}                 `json:"-"`
	XXX_unrecognized     []byte                   `json:"-"`
	XXX_sizecache        int32                    `json:"-"`
}

func (m *NodeExecStats) Reset()         { *m = NodeExecStats{} }
func (m *NodeExecStats) String() string { return proto.CompactTextString(m) }
func (*NodeExecStats) ProtoMessage()    {}
func (*NodeExecStats) Descriptor() ([]byte, []int) {
	return fileDescriptor_1e915309f7ed52e5, []int{2}
}

func (m *NodeExecStats) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_NodeExecStats.Unmarshal(m, b)
}
func (m *NodeExecStats) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_NodeExecStats.Marshal(b, m, deterministic)
}
func (m *NodeExecStats) XXX_Merge(src proto.Message) {
	xxx_messageInfo_NodeExecStats.Merge(m, src)
}
func (m *NodeExecStats) XXX_Size() int {
	return xxx_messageInfo_NodeExecStats.Size(m)
}
func (m *NodeExecStats) XXX_DiscardUnknown() {
	xxx_messageInfo_NodeExecStats.DiscardUnknown(m)
}

var xxx_messageInfo_NodeExecStats proto.InternalMessageInfo

func (m *NodeExecStats) GetNodeName() string {
	if m != nil {
		return m.NodeName
	}
	return ""
}

func (m *NodeExecStats) GetAllStartMicros() int64 {
	if m != nil {
		return m.AllStartMicros
	}
	return 0
}

func (m *NodeExecStats) GetOpStartRelMicros() int64 {
	if m != nil {
		return m.OpStartRelMicros
	}
	return 0
}

func (m *NodeExecStats) GetOpEndRelMicros() int64 {
	if m != nil {
		return m.OpEndRelMicros
	}
	return 0
}

func (m *NodeExecStats) GetAllEndRelMicros() int64 {
	if m != nil {
		return m.AllEndRelMicros
	}
	return 0
}

func (m *NodeExecStats) GetMemory() []*AllocatorMemoryUsed {
	if m != nil {
		return m.Memory
	}
	return nil
}

func (m *NodeExecStats) GetOutput() []*NodeOutput {
	if m != nil {
		return m.Output
	}
	return nil
}

func (m *NodeExecStats) GetTimelineLabel() string {
	if m != nil {
		return m.TimelineLabel
	}
	return ""
}

func (m *NodeExecStats) GetScheduledMicros() int64 {
	if m != nil {
		return m.ScheduledMicros
	}
	return 0
}

func (m *NodeExecStats) GetThreadId() uint32 {
	if m != nil {
		return m.ThreadId
	}
	return 0
}

func (m *NodeExecStats) GetReferencedTensor() []*AllocationDescription {
	if m != nil {
		return m.ReferencedTensor
	}
	return nil
}

type DeviceStepStats struct {
	Device               string           `protobuf:"bytes,1,opt,name=device,proto3" json:"device,omitempty"`
	NodeStats            []*NodeExecStats `protobuf:"bytes,2,rep,name=node_stats,json=nodeStats,proto3" json:"node_stats,omitempty"`
	XXX_NoUnkeyedLiteral struct{}         `json:"-"`
	XXX_unrecognized     []byte           `json:"-"`
	XXX_sizecache        int32            `json:"-"`
}

func (m *DeviceStepStats) Reset()         { *m = DeviceStepStats{} }
func (m *DeviceStepStats) String() string { return proto.CompactTextString(m) }
func (*DeviceStepStats) ProtoMessage()    {}
func (*DeviceStepStats) Descriptor() ([]byte, []int) {
	return fileDescriptor_1e915309f7ed52e5, []int{3}
}

func (m *DeviceStepStats) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_DeviceStepStats.Unmarshal(m, b)
}
func (m *DeviceStepStats) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_DeviceStepStats.Marshal(b, m, deterministic)
}
func (m *DeviceStepStats) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DeviceStepStats.Merge(m, src)
}
func (m *DeviceStepStats) XXX_Size() int {
	return xxx_messageInfo_DeviceStepStats.Size(m)
}
func (m *DeviceStepStats) XXX_DiscardUnknown() {
	xxx_messageInfo_DeviceStepStats.DiscardUnknown(m)
}

var xxx_messageInfo_DeviceStepStats proto.InternalMessageInfo

func (m *DeviceStepStats) GetDevice() string {
	if m != nil {
		return m.Device
	}
	return ""
}

func (m *DeviceStepStats) GetNodeStats() []*NodeExecStats {
	if m != nil {
		return m.NodeStats
	}
	return nil
}

type StepStats struct {
	DevStats             []*DeviceStepStats `protobuf:"bytes,1,rep,name=dev_stats,json=devStats,proto3" json:"dev_stats,omitempty"`
	XXX_NoUnkeyedLiteral struct{}           `json:"-"`
	XXX_unrecognized     []byte             `json:"-"`
	XXX_sizecache        int32              `json:"-"`
}

func (m *StepStats) Reset()         { *m = StepStats{} }
func (m *StepStats) String() string { return proto.CompactTextString(m) }
func (*StepStats) ProtoMessage()    {}
func (*StepStats) Descriptor() ([]byte, []int) {
	return fileDescriptor_1e915309f7ed52e5, []int{4}
}

func (m *StepStats) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_StepStats.Unmarshal(m, b)
}
func (m *StepStats) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_StepStats.Marshal(b, m, deterministic)
}
func (m *StepStats) XXX_Merge(src proto.Message) {
	xxx_messageInfo_StepStats.Merge(m, src)
}
func (m *StepStats) XXX_Size() int {
	return xxx_messageInfo_StepStats.Size(m)
}
func (m *StepStats) XXX_DiscardUnknown() {
	xxx_messageInfo_StepStats.DiscardUnknown(m)
}

var xxx_messageInfo_StepStats proto.InternalMessageInfo

func (m *StepStats) GetDevStats() []*DeviceStepStats {
	if m != nil {
		return m.DevStats
	}
	return nil
}

func init() {
	proto.RegisterType((*AllocatorMemoryUsed)(nil), "tensorflow.AllocatorMemoryUsed")
	proto.RegisterType((*NodeOutput)(nil), "tensorflow.NodeOutput")
	proto.RegisterType((*NodeExecStats)(nil), "tensorflow.NodeExecStats")
	proto.RegisterType((*DeviceStepStats)(nil), "tensorflow.DeviceStepStats")
	proto.RegisterType((*StepStats)(nil), "tensorflow.StepStats")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/step_stats.proto", fileDescriptor_1e915309f7ed52e5)
}

var fileDescriptor_1e915309f7ed52e5 = []byte{
	// 562 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x53, 0x4d, 0x6f, 0xda, 0x40,
	0x10, 0x95, 0xf3, 0x41, 0xf1, 0x20, 0x0a, 0x6c, 0xa4, 0xc8, 0x0d, 0x8a, 0x42, 0x91, 0x2a, 0x91,
	0x56, 0x85, 0x8a, 0x4a, 0x6d, 0xae, 0x45, 0xe1, 0x50, 0x29, 0xa1, 0x91, 0x69, 0xcf, 0xd6, 0xe2,
	0x1d, 0x1a, 0x2b, 0x6b, 0xaf, 0xb5, 0x5e, 0x48, 0x23, 0xf5, 0xda, 0xff, 0xdc, 0x63, 0xb5, 0x63,
	0x83, 0x5d, 0x92, 0xdc, 0xc6, 0x6f, 0xdf, 0x3c, 0xbf, 0xd9, 0x79, 0x0b, 0x6f, 0x0d, 0x26, 0x99,
	0xd2, 0x4b, 0xa9, 0xee, 0x47, 0xa1, 0xd2, 0x38, 0x5a, 0x6a, 0x1e, 0xe3, 0xbd, 0xd2, 0x77, 0xa3,
	0xcc, 0x60, 0x1a, 0x64, 0x86, 0x9b, 0x6c, 0x98, 0x6a, 0x65, 0x14, 0x83, 0x92, 0x7b, 0xf2, 0xe9,
	0xf9, 0x3e, 0x2e, 0xa5, 0x0a, 0xb9, 0x89, 0x54, 0x12, 0x08, 0xcc, 0x42, 0x1d, 0xa5, 0xb6, 0xce,
	0x35, 0x4e, 0xc6, 0xcf, 0xf7, 0xe5, 0x27, 0x8f, 0x7b, 0xfa, 0xbf, 0xe1, 0xe8, 0x4b, 0xae, 0xa9,
	0xf4, 0x35, 0xc6, 0x4a, 0x3f, 0xfc, 0xc8, 0x50, 0xb0, 0x37, 0xf0, 0x92, 0x6f, 0xe0, 0x20, 0xe1,
	0x31, 0x7a, 0x4e, 0xcf, 0x19, 0xb8, 0x7e, 0x73, 0x8b, 0xce, 0x78, 0x8c, 0xec, 0x0c, 0x1a, 0x46,
	0x19, 0x2e, 0x83, 0xc5, 0x83, 0xc1, 0xcc, 0xdb, 0xeb, 0x39, 0x83, 0x7d, 0x1f, 0x08, 0x9a, 0x58,
	0x84, 0x9d, 0x02, 0xa4, 0xc8, 0xef, 0x8a, 0xf3, 0x7d, 0x3a, 0x77, 0x2d, 0x42, 0xc7, 0xfd, 0x04,
	0x60, 0xa6, 0x04, 0x7e, 0x5b, 0x99, 0x74, 0x65, 0x18, 0x83, 0x83, 0x4c, 0x2a, 0x43, 0xbf, 0x3a,
	0xf4, 0xa9, 0x66, 0x57, 0xc0, 0x1e, 0x7b, 0x27, 0xa1, 0xc6, 0xf8, 0x74, 0x58, 0x0e, 0x3c, 0xfc,
	0x4e, 0xe5, 0x65, 0x49, 0xf2, 0x3b, 0x66, 0x17, 0xea, 0xff, 0x39, 0x80, 0xa6, 0xfd, 0xe1, 0xf4,
	0x17, 0x86, 0x73, 0x7b, 0xfb, 0xac, 0x0b, 0x6e, 0xa2, 0x04, 0x56, 0x67, 0xac, 0x5b, 0x80, 0xc6,
	0x1b, 0x40, 0x9b, 0x4b, 0x69, 0xf7, 0xa4, 0x4d, 0x10, 0x47, 0xa1, 0x56, 0x9b, 0x19, 0xed, 0xed,
	0xcc, 0x2d, 0x7c, 0x4d, 0x28, 0x7b, 0x0f, 0x47, 0x2a, 0x2d, 0x88, 0x1a, 0xe5, 0x86, 0x9c, 0x0f,
	0xdc, 0x56, 0x29, 0x71, 0x7d, 0x94, 0x05, 0xfd, 0x1c, 0x3a, 0x2a, 0x0d, 0x30, 0x11, 0x55, 0xf2,
	0x41, 0xae, 0xac, 0xd2, 0x69, 0x22, 0x4a, 0xea, 0x3b, 0x60, 0xd6, 0xc3, 0x0e, 0xf7, 0x90, 0xb8,
	0x2d, 0x2e, 0xe5, 0x7f, 0xe4, 0xcf, 0x50, 0x8b, 0x69, 0x89, 0x5e, 0xad, 0xb7, 0x3f, 0x68, 0x8c,
	0xcf, 0xaa, 0x37, 0xf4, 0xc4, 0x9e, 0xfd, 0x82, 0xce, 0x86, 0x50, 0x53, 0xb4, 0x04, 0xef, 0x05,
	0x35, 0x1e, 0x57, 0x1b, 0xcb, 0x15, 0xf9, 0x05, 0xcb, 0xe6, 0xc3, 0x44, 0x31, 0xca, 0x28, 0xc1,
	0x40, 0xf2, 0x05, 0x4a, 0xaf, 0x9e, 0xe7, 0x63, 0x83, 0x5e, 0x59, 0x90, 0x9d, 0x43, 0x3b, 0x0b,
	0x6f, 0x51, 0xac, 0x24, 0x8a, 0x8d, 0x75, 0x37, 0xb7, 0xbe, 0xc5, 0x0b, 0xeb, 0x5d, 0x70, 0xcd,
	0xad, 0x46, 0x2e, 0x82, 0x48, 0x78, 0xd0, 0x73, 0x06, 0x4d, 0xbf, 0x9e, 0x03, 0x5f, 0x05, 0x9b,
	0x41, 0x47, 0xe3, 0x12, 0x35, 0x26, 0x21, 0x8a, 0x20, 0xb7, 0xe6, 0x35, 0xc8, 0xe9, 0xeb, 0x27,
	0x46, 0x8c, 0x54, 0x52, 0x0d, 0x42, 0xbb, 0xec, 0xcd, 0x53, 0xd2, 0x0f, 0xa1, 0x75, 0x89, 0xeb,
	0x28, 0xc4, 0xb9, 0xc1, 0x34, 0x0f, 0xc2, 0x31, 0xd4, 0x04, 0x41, 0x45, 0x0a, 0x8a, 0x2f, 0x76,
	0x01, 0x40, 0x01, 0xa1, 0xc7, 0xea, 0xed, 0xd1, 0x3f, 0x5f, 0xed, 0xde, 0xce, 0x36, 0x4f, 0x3e,
	0xa5, 0x89, 0xca, 0xfe, 0x14, 0xdc, 0x52, 0xfe, 0x02, 0x5c, 0x81, 0xeb, 0x42, 0xc5, 0x21, 0x95,
	0x6e, 0x55, 0x65, 0xc7, 0x8e, 0x5f, 0x17, 0xb8, 0xa6, 0x6a, 0xf2, 0x01, 0x3c, 0xa5, 0x7f, 0x56,
	0xb9, 0xdb, 0x67, 0x3d, 0x69, 0x6d, 0x1b, 0x6e, 0xec, 0x6b, 0xce, 0x6e, 0x9c, 0xbf, 0x8e, 0xb3,
	0xa8, 0xd1, 0xd3, 0xfe, 0xf8, 0x2f, 0x00, 0x00, 0xff, 0xff, 0x9b, 0x01, 0xc5, 0x6d, 0x80, 0x04,
	0x00, 0x00,
}
