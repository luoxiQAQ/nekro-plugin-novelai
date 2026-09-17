# NovelAI 画图插件 (Nekro Agent)

NovelAI 文生图/图生图插件，适用于 [Nekro Agent](https://github.com/KroMiose/nekro-agent)。

## ✨ 功能

- **文生图** — `/画图 <描述>` 支持中文描述（自动翻译为 danbooru 标签）
- **图生图** — 通过 Sandbox Tool 由 LLM 代理调用
- **预设系统** — 人物预设 + 风格预设，支持直接写名称自动识别
- **智能翻译** — 按逗号分段翻译，预设展开后的英文标签不会被翻译器改动
- **重画** — `/重画` 使用上一次参数重新生成（不同随机种子）
- **元数据提取** — `/看参数` 回复图片提取 NovelAI PNG 元数据（自身生成的图片元数据已剥离）
- **WebUI 预设管理** — 暗色主题预设管理页面，支持增删改查
- **多 Token 轮切** — 支持多个 API Token 逗号分隔，自动轮切
- **R18 开关** — 关闭时自动添加 NSFW 负面提示词

## 📌 指令列表

| 指令 | 别名 | 说明 |
|------|------|------|
| `/画图 <描述>` | `nai`, `nai画图`, `novelai` | 文生图，支持中文（自动翻译） |
| `/重画 [追加描述]` | `redraw` | 用上一次参数重画 |
| `/看参数` | `反推`, `查看参数`, `naimeta` | 回复图片提取生成参数 |
| `/添加人物 <名称> <提示词>` | — | 添加/修改人物预设 |
| `/删除人物 <名称>` | — | 删除人物预设 |
| `/人物列表` | — | 查看所有人物预设 |
| `/添加风格 <名称> <提示词>` | — | 添加/修改风格预设 |
| `/删除风格 <名称>` | — | 删除风格预设 |
| `/风格列表` | — | 查看所有风格预设 |

## 🎭 预设使用

在画图描述中直接写预设名即可引用（也支持 `@人物名`、`#风格名` 语法）：

```
/画图 椿，花田，奔跑，横
/画图 @椿 #风格001 花田
```

仓库附带 `presets_example.json`，包含 58 个鸣潮（Wuthering Waves）角色预设，可导入使用。

## ⚙️ 配置

在 Nekro Agent 插件配置界面设置：

- **API Token** — NovelAI API Token（支持多个用逗号分隔）
- **模型** — 默认使用的 NAI 模型（v3/v4/v4.5/v5）
- **分辨率** — 默认图片分辨率
- **翻译模型** — 用于中文→英文标签翻译的聊天模型组
- **R18 开关** — 开启后允许生成 NSFW 内容

## 📦 安装

将本仓库克隆到 Nekro Agent 的 `plugins/packages/` 目录下：

```bash
cd /path/to/nekro_agent/plugins/packages/
git clone https://github.com/luoxiQAQ/nekro-plugin-novelai.git nekro_plugin_novelai
```

重启 Nekro Agent 即可。

## 📄 License

MIT
