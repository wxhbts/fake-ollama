# 使用官方 Go 镜像作为构建环境
FROM golang:1.21-alpine AS builder

# 设置工作目录
WORKDIR /app

# 复制 go.mod 和 go.sum 文件
COPY go.mod ./

# 下载依赖
RUN go mod download

# 复制源代码
COPY . .

# 构建应用
RUN CGO_ENABLED=0 GOOS=linux go build -o fake-ollama

# 使用轻量级的 alpine 镜像作为运行环境
FROM alpine:latest

# 安装必要的 CA 证书
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# 从构建阶段复制二进制文件
COPY --from=builder /app/fake-ollama .

# 暴露端口
EXPOSE 11434

# 运行应用
CMD ["./fake-ollama"] 