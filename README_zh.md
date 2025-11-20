# HALDxAI: Healthy Aging and Longevity Discovery AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-latest-brightgreen)](docs/)

HALDxAI是一个基于人工智能的健康衰老和长寿发现平台，旨在通过自然语言处理、机器学习和知识图谱技术，从科学文献中提取和分析衰老相关的实体、关系和模式。

## 🌟 主要功能

- **智能实体识别**: 使用LLM和SpaCy从生物医学文献中识别衰老相关实体
- **关系抽取**: 自动提取实体间的复杂关系网络
- **知识图谱构建**: 构建多维度衰老知识图谱
- **评分系统**: 基于多维度指标的实体和关系评分
- **可视化分析**: 丰富的网络可视化和分析工具
- **数据库集成**: 支持PostgreSQL和Neo4j数据库

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PostgreSQL 12+
- Neo4j 4.0+ (可选)
- 足够的计算资源用于LLM推理

### 安装

```bash
# 克隆仓库
git clone https://github.com/ShilongZhang116/haldxai.git
cd haldxai

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 安装包（开发模式）
pip install -e .
```

### 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量，添加API密钥等
nano .env

# 初始化项目配置
python -m haldxai.cli init
```

### 快速示例

```python
from haldxai import HALDxAI

# 初始化系统
hald = HALDxAI(config_path="configs/config.yaml")

# 运行NER管道
results = hald.run_ner_pipeline(
    input_file="data/raw/articles.csv",
    output_dir="data/processed/"
)

# 构建知识图谱
graph = hald.build_knowledge_graph(results)

# 生成可视化报告
hald.generate_report(graph, output_path="reports/analysis.html")
```

## 📁 项目结构

```
HALDxAI-Repository/
├── haldxai/                 # 主Python包
│   ├── core/               # 核心功能模块
│   ├── ner/                # 命名实体识别
│   ├── database/           # 数据库操作
│   ├── modeling/           # 机器学习模型
│   ├── scoring/            # 评分系统
│   ├── visualization/      # 可视化工具
│   └── workflow/           # 工作流管道
├── notebooks/              # 研究和分析notebooks
├── configs/                # 配置文件
├── data/                   # 数据目录
├── scripts/                # 实用脚本
├── tests/                  # 测试套件
└── docs/                   # 文档
```

## 📖 文档

- [安装指南](docs/installation.md)
- [使用教程](docs/usage.md)
- [API文档](docs/api/)
- [示例和案例研究](docs/examples/)

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定模块测试
pytest tests/test_core/

# 生成覆盖率报告
pytest --cov=haldxai tests/
```

## 🤝 贡献

我们欢迎社区贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目。

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 安装pre-commit钩子
pre-commit install

# 运行代码格式化
black haldxai/
isort haldxai/

# 运行类型检查
mypy haldxai/
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📊 引用

如果您在研究中使用了HALDxAI，请引用：

```bibtex
@software{haldxai2024,
  title={HALDxAI: Healthy Aging and Longevity Discovery AI},
  author={HALDxAI Development Team},
  year={2024},
  url={https://github.com/ShilongZhang116/haldxai}
}
```

## 🙏 致谢

感谢所有为HALDxAI项目做出贡献的研究者和开发者。

## 📞 联系我们

- 项目主页: https://github.com/ShilongZhang116/haldxai
- 问题反馈: https://github.com/ShilongZhang116/haldxai/issues
- 邮箱: shilongzhang@zju.edu.cn

## 使用HALDxAI在线服务：
- 线上服务：[https://bis.zju.edu.cn/haldxai](https://bis.zju.edu.cn/haldxai)

---

**注意**: 本项目仅用于研究目的，不提供医疗建议。