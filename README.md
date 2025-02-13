# Fake-Ollama

一个模拟 Ollama API 的服务器实现，Just for fun.

## 特性

- 完全兼容 Ollama API
- 支持 OpenAI 兼容接口
- 支持流式输出
- 模拟多个 DeepSeek-R1 模型
- 轻量级，易于部署

## 快速开始

### 使用 Docker

```bash
# 从 GitHub Container Registry 拉取
docker pull ghcr.io/spoonnotfound/fake-ollama:latest

# 运行容器
docker run -d -p 11434:11434 spoonnotfound/fake-ollama
```

### 从源码构建

1. 克隆仓库
```bash
git clone https://github.com/spoonnotfound/fake-ollama.git
cd fake-ollama
```

2. 构建
```bash
go build
```

3. 运行
```bash
./fake-ollama
```

## 配置

服务默认监听在 `0.0.0.0:11434`，可以通过环境变量修改配置：

```bash
# 修改服务地址和端口
export OLLAMA_HOST=0.0.0.0:11434
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License