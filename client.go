package tfclient

import (
	"context"
	"regexp"
	"strings"
	"sync"

	tfcore "github.com/joyang1/tfclient/tensorflow/core/framework"
	tf "github.com/joyang1/tfclient/tensorflow_serving"

	"google.golang.org/grpc"
)

type PredictionClient struct {
	mu      sync.RWMutex
	rpcConn *grpc.ClientConn
	svcConn tf.PredictionServiceClient
}

func NewClient(addr string) (*PredictionClient, error) {
	conn, err := grpc.Dial(addr, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	c := tf.NewPredictionServiceClient(conn)
	return &PredictionClient{rpcConn: conn, svcConn: c}, nil
}

func (c *PredictionClient) Predict(doc string) ([]float32, error) {
	maxSentence := 30
	maxDocLen := 650

	docValue, sentences, len := generateSimple(doc, maxSentence, maxDocLen)
	var document [][]byte
	for _, s := range docValue {
		for _, w := range s {
			document = append(document, []byte(w))
		}
	}

	resp, err := c.svcConn.Predict(context.Background(), &tf.PredictRequest{
		ModelSpec: &tf.ModelSpec{
			Name:          "note_quality",
			SignatureName: "predict_documents",
		},
		Inputs: map[string]*tfcore.TensorProto{
			"documents": &tfcore.TensorProto{
				Dtype:     tfcore.DataType_DT_STRING,
				StringVal: document,
				TensorShape: &tfcore.TensorShapeProto{
					Dim: []*tfcore.TensorShapeProto_Dim{
						&tfcore.TensorShapeProto_Dim{Size: 1},
						&tfcore.TensorShapeProto_Dim{Size: int64(maxDocLen)},
						&tfcore.TensorShapeProto_Dim{Size: int64(maxSentence)},
					},
				},
			},
			"sentence_lengths": &tfcore.TensorProto{
				Dtype:  tfcore.DataType_DT_INT32,
				IntVal: sentences,
				TensorShape: &tfcore.TensorShapeProto{
					Dim: []*tfcore.TensorShapeProto_Dim{
						&tfcore.TensorShapeProto_Dim{Size: 1},
						&tfcore.TensorShapeProto_Dim{Size: int64(maxDocLen)},
					},
				},
			},
			"document_lengths": &tfcore.TensorProto{
				Dtype:  tfcore.DataType_DT_INT32,
				IntVal: []int32{int32(len)},
				TensorShape: &tfcore.TensorShapeProto{
					Dim: []*tfcore.TensorShapeProto_Dim{
						&tfcore.TensorShapeProto_Dim{Size: 1},
					},
				},
			},
		},
	})
	if err != nil {
		return nil, err
	}

	respone := resp.Outputs["result"]
	result := respone.FloatVal
	return result, nil
}

func (c *PredictionClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.svcConn = nil
	return c.rpcConn.Close()
}

func generateSimple(document string, maxSentence int, maxDocLen int) ([][]string, []int32, int) {
	var docArr [][]string
	var sentenceLens []int32
	var sentences []string

	tb := strings.Split(document, "<title>")
	t := tb[0]
	b := tb[1]
	reg := regexp.MustCompile("[.。!！?？;；，]")
	sentences = append(sentences, t)
	sentences = append(sentences, reg.Split(b, -1)...)

	for _, s := range sentences {
		words, len := generateSegement(s, maxSentence)
		docArr = append(docArr, words)
		sentenceLens = append(sentenceLens, int32(len))
	}

	length := len(docArr)

	if length >= maxDocLen {
		docArr = docArr[0:maxDocLen]
		length = maxDocLen
	} else {
		var fill []string
		for i := 0; i < 30; i++ {
			fill = append(fill, "<P_Z>")
		}
		for i := length; i < maxDocLen; i++ {
			docArr = append(docArr, fill)
			sentenceLens = append(sentenceLens, 0)
		}
	}

	return docArr, sentenceLens, length
}

func generateSegement(body string, maxSentence int) ([]string, int) {
	sentences := strings.Split(body, " ")
	var words []string
	for _, s := range sentences {
		if len(strings.TrimSpace(s)) == 0 || len(s) == 0 {
			continue
		}
		words = append(words, s)
	}

	length := len(words)

	if length >= maxSentence {
		words = words[0:maxSentence]
		length = maxSentence
	} else {
		for i := length; i < maxSentence; i++ {
			words = append(words, "<P_Z>")
		}
	}

	return words, length
}
