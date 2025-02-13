package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
)

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   *bool     `json:"stream,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// 添加新的响应结构体
type ChatCompletionDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type ChatCompletionChoice struct {
	Index        int                `json:"index"`
	Delta        ChatCompletionDelta `json:"delta"`
	FinishReason *string            `json:"finish_reason"`
}

type ChatResponse struct {
	ID                string                `json:"id"`
	Object            string                `json:"object"`
	Created           int64                 `json:"created"`
	Model             string                `json:"model"`
	SystemFingerprint string                `json:"system_fingerprint"`
	Choices           []ChatCompletionChoice `json:"choices"`
}

type GenerateRequest struct {
	Model   string `json:"model"`
	Prompt  string `json:"prompt"`
	System  string `json:"system,omitempty"`
	Stream  *bool  `json:"stream,omitempty"`
	Raw     bool   `json:"raw,omitempty"`
	Format  string `json:"format,omitempty"`
}

type GenerateResponse struct {
	Model      string    `json:"model"`
	CreatedAt  time.Time `json:"created_at"`
	Response   string    `json:"response"`
	Done       bool      `json:"done"`
	Context    []int     `json:"context,omitempty"`
	TotalDuration    time.Duration `json:"total_duration,omitempty"`
	LoadDuration     time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount  int           `json:"prompt_eval_count,omitempty"`
	EvalCount        int           `json:"eval_count,omitempty"`
	DoneReason      string        `json:"done_reason,omitempty"`
}

// ModelDetails 存储模型的详细信息
type ModelDetails struct {
	Format            string   `json:"format,omitempty"`
	Family            string   `json:"family,omitempty"`
	Families          []string `json:"families,omitempty"`
	ParameterSize     string   `json:"parameter_size,omitempty"`
	QuantizationLevel string   `json:"quantization_level,omitempty"`
}

// ListModelResponse 表示单个模型的响应
type ListModelResponse struct {
	Model      string       `json:"model"`
	Name       string       `json:"name"`
	Size       int64        `json:"size"`
	Digest     string       `json:"digest"`
	ModifiedAt time.Time    `json:"modified_at"`
	Details    ModelDetails `json:"details"`
}

// ListResponse 表示模型列表的响应
type ListResponse struct {
	Models []ListModelResponse `json:"models"`
}

type PullRequest struct {
	Name     string `json:"name"`
	Stream   *bool  `json:"stream,omitempty"`
	Insecure bool   `json:"insecure,omitempty"`
}

type PullResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

type EmbedRequest struct {
	Model    string      `json:"model"`
	Input    interface{} `json:"input"`
	Truncate *bool       `json:"truncate,omitempty"`
}

type EmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
}

type CreateRequest struct {
	Name     string `json:"name"`
	Path     string `json:"path"`
	Stream   *bool  `json:"stream,omitempty"`
	Insecure bool   `json:"insecure,omitempty"`
}

type ShowRequest struct {
	Name    string `json:"name"`
	System  string `json:"system,omitempty"`
	Model   string `json:"model,omitempty"`
	Verbose bool   `json:"verbose,omitempty"`
}

type ShowResponse struct {
	License     string    `json:"license"`
	System      string    `json:"system"`
	Template    string    `json:"template"`
	Parameters  string    `json:"parameters"`
	ModelInfo   map[string]interface{} `json:"model_info"`
	ModifiedAt  time.Time `json:"modified_at"`
}

type CopyRequest struct {
	Source      string `json:"source"`
	Destination string `json:"destination"`
}

type DeleteRequest struct {
	Name  string `json:"name"`
	Model string `json:"model,omitempty"`
}

// 添加新的 OpenAI 兼容的响应结构体
type OpenAIModel struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Created   int64  `json:"created"`
	OwnedBy   string `json:"owned_by"`
}

type OpenAIModelList struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

// 添加全局变量来存储模型修改时间
var modelModifiedTimes map[string]time.Time

// 添加常量定义
const (
	// 中文响应文本
	ChineseThinkText = "嗯，我要开始思考了。"
	ChineseResponseText = "这是一条来自[fake-ollama](https://github.com/spoonnotfound/fake-ollama)的固定回复。"

	// 英文响应文本
	EnglishThinkText = "Hmm, let me think about it."
	EnglishResponseText = "This is a fixed response from [fake-ollama](https://github.com/spoonnotfound/fake-ollama)."
)

// 添加初始化函数
func init() {
	// 设置随机种子
	rand.Seed(time.Now().UnixNano())
	
	// 初始化时间映射
	baseTime := time.Now().UTC()
	modelModifiedTimes = make(map[string]time.Time)
	
	// 只保留 14b 及以上的模型
	models := []string{
		"deepseek-r1:14b",
		"deepseek-r1:32b",
		"deepseek-r1:70b",
		"deepseek-r1:671b",
	}
	
	for _, model := range models {
		offset := time.Duration(rand.Int63n(24*60*60)) * time.Second
		modelModifiedTimes[model] = baseTime.Add(-offset)
	}
}

// 修改辅助函数为按字符分割
func splitIntoChunks(text string) []string {
	var chunks []string
	for _, r := range text {
		chunks = append(chunks, string(r))
	}
	return chunks
}

// 修改语言检测函数
func containsEnglish(str string) bool {
	for _, r := range str {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			return true
		}
	}
	return false
}

// 修改获取响应文本的函数
func getResponseTexts(input string) (thinkText, responseText string) {
	if !containsEnglish(input) { // 如果不包含英文字母，则返回中文
		return ChineseThinkText, ChineseResponseText
	}
	return EnglishThinkText, EnglishResponseText
}

// 添加辅助函数来验证模型
func isValidModel(model string) bool {
	validModels := map[string]bool{
		"deepseek-r1:14b":  true,
		"deepseek-r1:32b":  true,
		"deepseek-r1:70b":  true,
		"deepseek-r1:671b": true,
	}
	return validModels[model]
}

// 添加从 openai/middleware.go 移动过来的错误类型
type Error struct {
	Message string      `json:"message"`
	Type    string      `json:"type"`
	Param   interface{} `json:"param"`
	Code    *string     `json:"code"`
}

type ErrorResponse struct {
	Error Error `json:"error"`
}

// 添加错误创建函数
func NewError(code int, message string) ErrorResponse {
	var etype string
	switch code {
	case http.StatusBadRequest:
		etype = "invalid_request_error"
	case http.StatusNotFound:
		etype = "not_found_error"
	default:
		etype = "api_error"
	}

	return ErrorResponse{Error{Type: etype, Message: message}}
}

// 添加中间件类型和函数
type bodyLogWriter struct {
	gin.ResponseWriter
	body *bytes.Buffer
}

func (w bodyLogWriter) Write(b []byte) (int, error) {
	return w.body.Write(b)  // 只写入到缓冲区，不写入到响应
}

// 完整实现中间件函数
func ListMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 保存原始 Writer
		originalWriter := c.Writer
		
		// 创建自定义 ResponseWriter 来捕获响应
		blw := &bodyLogWriter{
			body:           bytes.NewBufferString(""),
			ResponseWriter: c.Writer,
		}
		c.Writer = blw
		
		c.Next()
		
		// 恢复原始 Writer
		c.Writer = originalWriter

		// 解析原始响应
		var listResponse ListResponse
		if err := json.Unmarshal(blw.body.Bytes(), &listResponse); err != nil {
			c.JSON(http.StatusInternalServerError, NewError(http.StatusInternalServerError, "invalid response format"))
			return
		}

		// 转换为 OpenAI 格式
		models := make([]OpenAIModel, len(listResponse.Models))
		for i, m := range listResponse.Models {
			models[i] = OpenAIModel{
				ID:        m.Name,
				Object:    "model",
				Created:   m.ModifiedAt.Unix(),
				OwnedBy:   "library",
			}
		}

		response := OpenAIModelList{
			Object: "list",
			Data:   models,
		}

		// 写入转换后的响应
		c.JSON(http.StatusOK, response)
	}
}

func ChatMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 生成唯一ID
		id := fmt.Sprintf("chatcmpl-%d", rand.Intn(999))
		c.Set("chat_id", id)
		c.Next()
	}
}

func CompletionsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 生成唯一ID
		id := fmt.Sprintf("cmpl-%d", rand.Intn(999))
		c.Set("completion_id", id)
		c.Next()
	}
}

func EmbeddingsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			Input any    `json:"input"`
			Model string `json:"model"`
		}
		
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// 验证输入
		if req.Input == nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError(http.StatusBadRequest, "input is required"))
			return
		}

		c.Next()
	}
}

func RetrieveMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		modelName := c.Param("model")
		
		// 构造请求体
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(map[string]string{
			"name": modelName,
		}); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		// 替换请求体
		c.Request.Body = io.NopCloser(&b)
		
		c.Next()
	}
}

// 添加 SSE 响应相关的辅助函数
func sendSSEResponse(c *gin.Context, data interface{}) {
	c.SSEvent("", data)
	c.Writer.Flush()
}

func setupSSEHeaders(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
}

// 创建聊天响应的辅助函数
func createChatResponse(id string, timestamp int64, model string, delta ChatCompletionDelta, finishReason *string) ChatResponse {
	return ChatResponse{
		ID:                id,
		Object:            "chat.completion.chunk",
		Created:           timestamp,
		Model:             model,
		SystemFingerprint: "fp_ollama",
		Choices: []ChatCompletionChoice{
			{
				Index:        0,
				Delta:        delta,
				FinishReason: finishReason,
			},
		},
	}
}

// 根据模型获取延迟时间
func getModelDelay(model string) time.Duration {
	switch model {
	case "deepseek-r1:671b":
		return 200 * time.Millisecond
	case "deepseek-r1:70b":
		return 100 * time.Millisecond
	case "deepseek-r1:32b":
		return 50 * time.Millisecond
	default:
		return 30 * time.Millisecond
	}
}

// 发送思考过程
func sendThinkingProcess(c *gin.Context, id string, timestamp int64, model, thinkText string, delay time.Duration) {
	// 发送开始标记
	startResp := createChatResponse(id, timestamp, model, 
		ChatCompletionDelta{Role: "assistant", Content: "<think>"}, nil)
	sendSSEResponse(c, startResp)
	time.Sleep(delay)

	// 发送换行
	newlineResp := createChatResponse(id, timestamp, model,
		ChatCompletionDelta{Content: "\n"}, nil)
	sendSSEResponse(c, newlineResp)
	time.Sleep(delay)

	// 发送3遍思考内容
	for i := 0; i < 3; i++ {
		chunks := splitIntoChunks(thinkText)
		for _, chunk := range chunks {
			resp := createChatResponse(id, timestamp, model,
				ChatCompletionDelta{Content: chunk}, nil)
			sendSSEResponse(c, resp)
			time.Sleep(delay)
		}
	}

	// 发送结束标记
	endResp := createChatResponse(id, timestamp, model,
		ChatCompletionDelta{Content: "</think>"}, nil)
	sendSSEResponse(c, endResp)
	time.Sleep(delay)

	// 发送换行
	doubleNewlineResp := createChatResponse(id, timestamp, model,
		ChatCompletionDelta{Content: "\n\n"}, nil)
	sendSSEResponse(c, doubleNewlineResp)
	time.Sleep(delay)
}

func chatHandler(c *gin.Context) {
	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, NewError(http.StatusBadRequest, err.Error()))
		return
	}

	// 验证模型
	if !isValidModel(req.Model) {
		c.JSON(http.StatusNotFound, NewError(
			http.StatusNotFound,
			fmt.Sprintf("model %q not found, try pulling it first", req.Model),
		))
		return
	}

	// 验证消息
	if req.Messages == nil || len(req.Messages) == 0 {
		c.JSON(http.StatusBadRequest, NewError(http.StatusBadRequest, "messages array is required"))
		return
	}

	lastMsg := req.Messages[len(req.Messages)-1]
	if lastMsg.Role == "" || lastMsg.Content == "" {
		c.JSON(http.StatusBadRequest, NewError(http.StatusBadRequest, "invalid message format"))
		return
	}

	thinkText, responseText := getResponseTexts(lastMsg.Content)
	delay := getModelDelay(req.Model)

	if req.Stream != nil && *req.Stream {
		setupSSEHeaders(c)
		
		id := fmt.Sprintf("chatcmpl-%d", rand.Intn(999))
		timestamp := time.Now().Unix()

		// 发送思考过程
		sendThinkingProcess(c, id, timestamp, req.Model, thinkText, delay)

		// 发送响应内容
		chunks := splitIntoChunks(responseText)
		for i, chunk := range chunks {
			select {
			case <-c.Request.Context().Done():
				return
			default:
				var finishReason *string
				if i == len(chunks)-1 {
					s := "stop"
					finishReason = &s
				}
				resp := createChatResponse(id, timestamp, req.Model,
					ChatCompletionDelta{Content: chunk}, finishReason)
				sendSSEResponse(c, resp)
				time.Sleep(delay)
			}
		}
		return
	}

	// 非流式请求
	response := createChatResponse(fmt.Sprintf("chatcmpl-%d", time.Now().Unix()), time.Now().Unix(), req.Model, ChatCompletionDelta{
		Role:    "assistant",
		Content: responseText,
	}, func() *string { s := "stop"; return &s }())

	c.JSON(http.StatusOK, response)
}

func generateHandler(c *gin.Context) {
	var req GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, NewError(http.StatusBadRequest, err.Error()))
		return
	}

	// 验证模型
	if !isValidModel(req.Model) {
		c.JSON(http.StatusNotFound, NewError(
			http.StatusNotFound,
			fmt.Sprintf("model %q not found, try pulling it first", req.Model),
		))
		return
	}

	if req.Prompt == "" {
		// 如果没有提示词，返回加载成功响应
		c.JSON(http.StatusOK, GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	// 根据输入内容判断语言
	isChinese := containsEnglish(req.Prompt)
	
	var responseText string
	if isChinese {
		responseText = ChineseResponseText
	} else {
		responseText = EnglishResponseText
	}

	startTime := time.Now()
	loadDuration := 100 * time.Millisecond // 模拟加载时间

	// 根据模型大小设置延迟
	var delay time.Duration
	delay = getModelDelay(req.Model)

	if req.Stream != nil && *req.Stream {
		setupSSEHeaders(c)
		
		chunks := splitIntoChunks(responseText)
		for i, chunk := range chunks {
			select {
			case <-c.Request.Context().Done():
				return
			default:
				partialResponse := GenerateResponse{
					Model:     req.Model,
					CreatedAt: time.Now().UTC(),
					Response:  chunk,
					Done:     i == len(chunks)-1,
				}
				
				if i == len(chunks)-1 {
					partialResponse.TotalDuration = time.Since(startTime)
					partialResponse.LoadDuration = loadDuration
					partialResponse.PromptEvalCount = 10
					partialResponse.EvalCount = 20
					partialResponse.Context = []int{1, 2, 3}
				}
				
				sendSSEResponse(c, partialResponse)
				time.Sleep(delay)
			}
		}
		return
	}

	response := GenerateResponse{
		Model:           req.Model,
		CreatedAt:       time.Now().UTC(),
		Response:        responseText,
		Done:           true,
		Context:        []int{1, 2, 3}, // 模拟上下文token
		TotalDuration:  time.Since(startTime),
		LoadDuration:   loadDuration,
		PromptEvalCount: 10, // 模拟评估计数
		EvalCount:       20,
	}

	c.JSON(http.StatusOK, response)
}

func ListHandler(c *gin.Context) {
	// 检查客户端连接是否已经关闭
	if c.Writer.Written() {
		return
	}

	// 创建所有 DeepSeek-R1 模型数据（只保留 14b 及以上）
	models := []ListModelResponse{
		{
			Model:      "deepseek-r1:14b",
			Name:       "deepseek-r1:14b",
			Size:       9663676416, // 约 9.0GB
			Digest:     "c42c25d8c10a841bd24724309898ae851466696a7d7f3a0a408b895538ccbc98",
			ModifiedAt: modelModifiedTimes["deepseek-r1:14b"],
			Details: ModelDetails{
				Format:            "gguf",
				Family:            "qwen2",
				Families:          []string{"qwen2"},
				ParameterSize:     "14B",
				QuantizationLevel: "Q4_K_M",
			},
		},
		{
			Model:      "deepseek-r1:32b",
			Name:       "deepseek-r1:32b",
			Size:       21474836480, // 约 20GB
			Digest:     "d42c25d8c10a841bd24724309898ae851466696a7d7f3a0a408b895538ccbc99",
			ModifiedAt: modelModifiedTimes["deepseek-r1:32b"],
			Details: ModelDetails{
				Format:            "gguf",
				Family:            "qwen2",
				Families:          []string{"qwen2"},
				ParameterSize:     "32B",
				QuantizationLevel: "Q4_K_M",
			},
		},
		{
			Model:      "deepseek-r1:70b",
			Name:       "deepseek-r1:70b",
			Size:       46170898432, // 约 43GB
			Digest:     "e42c25d8c10a841bd24724309898ae851466696a7d7f3a0a408b895538ccbc9a",
			ModifiedAt: modelModifiedTimes["deepseek-r1:70b"],
			Details: ModelDetails{
				Format:            "gguf",
				Family:            "llama",
				Families:          []string{"llama"},
				ParameterSize:     "70B",
				QuantizationLevel: "Q4_K_M",
			},
		},
		{
			Model:      "deepseek-r1:671b",
			Name:       "deepseek-r1:671b",
			Size:       433791590400, // 约 404GB
			Digest:     "f42c25d8c10a841bd24724309898ae851466696a7d7f3a0a408b895538ccbc9b",
			ModifiedAt: modelModifiedTimes["deepseek-r1:671b"],
			Details: ModelDetails{
				Format:            "gguf",
				Family:            "deepseek",
				Families:          []string{"deepseek"},
				ParameterSize:     "671B",
				QuantizationLevel: "Q4_K_M",
			},
		},
	}

	// 设置响应头
	c.Header("Content-Type", "application/json")
	c.Header("Connection", "keep-alive")

	select {
	case <-c.Request.Context().Done():
		// 客户端已断开连接
		return
	default:
		// 返回模型列表
		c.JSON(http.StatusOK, ListResponse{
			Models: models,
		})
	}
}

func pullHandler(c *gin.Context) {
	var req PullRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Stream != nil && *req.Stream {
		setupSSEHeaders(c)
		
		total := int64(1000)
		for completed := int64(0); completed <= total; completed += 100 {
			response := PullResponse{
				Status:    "downloading model",
				Digest:    "sha256:fake789",
				Total:     total,
				Completed: completed,
			}
			if completed == total {
				response.Status = "success"
			}
			sendSSEResponse(c, response)
			time.Sleep(500 * time.Millisecond)
		}
		return
	}

	response := PullResponse{
		Status: "success",
		Digest: "sha256:fake789",
	}
	c.JSON(http.StatusOK, response)
}

func embedHandler(c *gin.Context) {
	var req EmbedRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 将输入转换为字符串数组
	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []interface{}:
		for _, item := range v {
			if str, ok := item.(string); ok {
				inputs = append(inputs, str)
			}
		}
	}

	// 生成假的embeddings
	embeddings := make([][]float32, len(inputs))
	for i := range inputs {
		embeddings[i] = make([]float32, 4096) // 假设是4096维的向量
		for j := range embeddings[i] {
			embeddings[i][j] = 0.1 // 所有维度设置为0.1
		}
	}

	response := EmbedResponse{
		Model:      req.Model,
		Embeddings: embeddings,
	}

	c.JSON(http.StatusOK, response)
}

func createHandler(c *gin.Context) {
	var req CreateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Stream != nil && *req.Stream {
		setupSSEHeaders(c)
		
		steps := []string{
			"parsing modelfile",
			"creating model",
			"completed",
		}

		for _, step := range steps {
			sendSSEResponse(c, gin.H{
				"status": step,
				"digest": "sha256:fake789",
			})
			time.Sleep(500 * time.Millisecond)
		}
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"digest": "sha256:fake789",
	})
}

func showHandler(c *gin.Context) {
	var req ShowRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	response := ShowResponse{
		License:  "MIT",
		System:   req.System,
		Template: "{{ .Prompt }}",
		Parameters: "temperature 0.7\ntop_k 50\ntop_p 0.9",
		ModelInfo: map[string]interface{}{
			"format": "gguf",
			"family": "llama",
			"families": []string{"llama", "llama2"},
			"parameter_size": "7B",
			"quantization_level": "Q4_K_M",
		},
		ModifiedAt: time.Now().UTC(),
	}

	c.JSON(http.StatusOK, response)
}

func copyHandler(c *gin.Context) {
	var req CopyRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
	})
}

func deleteHandler(c *gin.Context) {
	var req DeleteRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
	})
}

func versionHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"version": "0.5.7",
	})
}

func healthHandler(c *gin.Context) {
	c.String(http.StatusOK, "Ollama is running")
}

func main() {
	r := gin.Default()

	// 添加全局错误处理中间件
	r.Use(func(c *gin.Context) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("Panic recovered: %v", err)
				if !c.Writer.Written() {
					c.AbortWithStatus(http.StatusInternalServerError)
				}
			}
		}()
		c.Next()
	})

	// 设置CORS
	r.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	// 基础API路由
	r.POST("/api/chat", chatHandler)
	r.POST("/api/generate", generateHandler)
	r.GET("/api/tags", ListHandler)
	r.POST("/api/pull", pullHandler)
	r.POST("/api/embed", embedHandler)
	r.POST("/api/embeddings", embedHandler) // 兼容性接口
	r.POST("/api/create", createHandler)
	r.POST("/api/show", showHandler)
	r.POST("/api/copy", copyHandler)
	r.DELETE("/api/delete", deleteHandler)
	r.POST("/api/delete", deleteHandler) // 兼容性接口

	// OpenAI 兼容性路由
	r.POST("/v1/chat/completions", ChatMiddleware(), chatHandler)
	r.POST("/v1/completions", CompletionsMiddleware(), generateHandler)
	r.POST("/v1/embeddings", EmbeddingsMiddleware(), embedHandler)
	r.GET("/v1/models", ListMiddleware(), ListHandler)
	r.GET("/v1/models/:model", RetrieveMiddleware(), showHandler)

	// 健康检查和版本信息
	r.GET("/", healthHandler)
	r.HEAD("/", healthHandler)
	r.GET("/api/version", versionHandler)
	r.HEAD("/api/version", versionHandler)

	// 获取服务地址配置
	addr := os.Getenv("OLLAMA_HOST")
	if addr == "" {
		addr = "0.0.0.0:11434"
	}

	// 启动服务器
	log.Printf("Starting fake-ollama server on %s...", addr)
	if err := r.Run(addr); err != nil {
		log.Fatal(err)
	}
}
