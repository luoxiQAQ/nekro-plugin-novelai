import asyncio
import time as _time
import base64
import io
import json
import random
import re
import threading
import zipfile
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import Field

from nekro_agent.api import i18n
from nekro_agent.api.plugin import (
    Arg,
    CmdCtl,
    CommandExecutionContext,
    CommandPermission,
    CommandResponse,
    ConfigBase,
    ExtraField,
    NekroPlugin,
    SandboxMethodType,
)
from nekro_agent.services.command.schemas import (
    CommandOutputSegment,
    CommandOutputSegmentType,
)
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import logger


plugin = NekroPlugin(
    name="NovelAI 画图",
    module_name="novelai",
    description="NovelAI 文生图/图生图插件，支持 NAI v3/v4/v4.5/v5 模型。",
    version="1.2.0",
    author="luoxi",
    url="",
    i18n_name=i18n.i18n_text(zh_CN="NovelAI 画图", en_US="NovelAI Image"),
    i18n_description=i18n.i18n_text(
        zh_CN="NovelAI 文生图/图生图插件，支持 NAI v3/v4/v4.5/v5 模型，内置翻译和 R18 开关。",
        en_US="NovelAI text-to-image and image-to-image plugin with translation and R18 toggle.",
    ),
    webui_path="/",
    allow_sleep=True,
    sleep_brief="提供 NovelAI 文生图、图生图能力。仅在用户明确要求画图、生成图片时激活。",
)

NAI_API_BASE = "https://image.novelai.net"

MODEL_CHOICES = Literal[
    "nai-diffusion-5-full",
    "nai-diffusion-5-curated",
    "nai-diffusion-4-5-full",
    "nai-diffusion-4-5-curated",
]

RESOLUTION_CHOICES = Literal[
    "832x1216",
    "1216x832",
    "1024x1024",
]

V4_STYLE_MODELS = {
    "nai-diffusion-4", "nai-diffusion-4-curated-preview",
    "nai-diffusion-4.5-full", "nai-diffusion-4-5-full", "nai-diffusion-4-5-curated",
    "nai-diffusion-5-full", "nai-diffusion-5-curated",
}


def _is_v4_style(model: str) -> bool:
    return model.lower() in V4_STYLE_MODELS


def _model_label(model: str) -> str:
    ml = model.lower()
    if "5-full" in ml or "5-curated" in ml:
        return "v5"
    if "4-5" in ml or "4.5" in ml:
        return "v4.5"
    if "4" in ml:
        return "v4"
    return "v3"


def _create_v4_prompt(prompt: str) -> dict:
    return {
        "caption": {"base_caption": prompt, "char_captions": [], "scenery_captions": []},
        "use_coords": False,
        "use_order": True,
    }


@plugin.mount_config()
class NovelAIConfig(ConfigBase):
    API_TOKEN: str = Field(
        default="",
        title="NovelAI API Token",
        description="NovelAI API Token，支持多个用逗号分隔，自动轮切。",
        json_schema_extra=ExtraField(is_secret=True).model_dump(),
    )
    API_PROXY: str = Field(
        default="",
        title="API 反代地址",
        description="NovelAI API 反向代理地址（替换 https://image.novelai.net），留空使用官方地址。",
    )
    HTTP_PROXY: str = Field(
        default="",
        title="HTTP 代理",
        description="HTTP/SOCKS5 代理地址，留空不使用代理。",
    )
    DEFAULT_MODEL: MODEL_CHOICES = Field(
        default="nai-diffusion-4-5-full",
        title="模型",
        description="默认使用的 NovelAI 模型。",
    )
    DEFAULT_RESOLUTION: RESOLUTION_CHOICES = Field(
        default="832x1216",
        title="分辨率",
        description="默认图片分辨率。",
    )
    DEFAULT_STEPS: int = Field(default=28, title="步数 (1-50)", description="默认采样步数。", ge=1, le=50)
    DEFAULT_SCALE: int = Field(default=7, title="权重 (1-20)", description="默认 CFG Scale 引导权重。", ge=1, le=20)
    CFG_RESCALE: float = Field(default=0.0, title="引导缩放参数(0.0-1.0)", description="CFG Rescale 参数。", ge=0.0, le=1.0)
    DEFAULT_SAMPLER: str = Field(default="k_euler", title="默认采样器", description="默认采样器名称。")
    NEGATIVE_PROMPT: str = Field(
        default="",
        title="负面提示词",
        description="全局负面提示词，留空使用默认。",
        json_schema_extra=ExtraField(is_textarea=True).model_dump(),
    )
    IMG2IMG_STRENGTH: float = Field(default=0.5, title="图生图强度", description="图生图强度 (0.1-0.9)。", ge=0.1, le=0.9)
    IMG2IMG_NOISE: float = Field(default=0.2, title="图生图噪声", description="图生图噪声 (0.0-1.0)。", ge=0.0, le=1.0)
    ENABLE_R18: bool = Field(
        default=False,
        title="R18 画图开关",
        description="开启后允许生成 R18 内容。关闭时自动在负面提示词中添加 NSFW 过滤标签。",
    )
    TRANSLATE_MODEL_GROUP: str = Field(
        default="",
        title="翻译大模型",
        description="用于将中文提示词翻译为英文 danbooru 标签的聊天模型组。留空则不翻译。",
        json_schema_extra=ExtraField(ref_model_groups=True, required=False, model_type="chat").model_dump(),
    )


config: NovelAIConfig = plugin.get_config(NovelAIConfig)

_token_index = 0

PRESET_KIND_LABELS = {"characters": "人物", "styles": "风格"}


class PresetStore:
    def __init__(self) -> None:
        self.path = plugin.get_plugin_data_dir() / "presets.json"
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            data = {}
        return {
            "characters": self._clean_map(data.get("characters", {})),
            "styles": self._clean_map(data.get("styles", {})),
        }

    @staticmethod
    def _clean_map(value: Any) -> dict:
        if not isinstance(value, dict):
            return {}
        return {
            str(name).strip(): str(prompt).strip()
            for name, prompt in value.items()
            if str(name).strip() and str(prompt).strip()
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.path)

    @staticmethod
    def _check_kind(kind: str) -> str:
        if kind not in PRESET_KIND_LABELS:
            raise ValueError("预设类型必须是 characters 或 styles")
        return kind

    @staticmethod
    def _check_values(name: str, prompt: str) -> tuple[str, str]:
        name = name.strip()
        prompt = prompt.strip()
        if not name or not prompt:
            raise ValueError("预设名称和提示词都不能为空")
        if re.search(r"[\s@#=<>/'\"]", name):
            raise ValueError("预设名称不能包含空格、@、#、=、尖括号、斜杠或引号")
        if len(name) > 40:
            raise ValueError("预设名称不能超过 40 个字符")
        if len(prompt) > 4000:
            raise ValueError("预设提示词不能超过 4000 个字符")
        return name, prompt

    def all(self) -> dict:
        with self._lock:
            return {kind: dict(values) for kind, values in self._data.items()}

    def get(self, kind: str, name: str) -> Optional[str]:
        kind = self._check_kind(kind)
        with self._lock:
            return self._data[kind].get(name.strip())

    def set(self, kind: str, name: str, prompt: str) -> None:
        kind = self._check_kind(kind)
        name, prompt = self._check_values(name, prompt)
        with self._lock:
            self._data[kind][name] = prompt
            self._save()

    def delete(self, kind: str, name: str) -> bool:
        kind = self._check_kind(kind)
        with self._lock:
            if name.strip() not in self._data[kind]:
                return False
            del self._data[kind][name.strip()]
            self._save()
            return True


preset_store = PresetStore()

_last_draw: dict[str, dict] = {}


def _expand_preset_prompt(prompt: str) -> str:
    character_names: list[str] = []
    style_names: list[str] = []

    def collect(kind: str, names: list[str], name: str) -> str:
        name = name.strip()
        if name and name not in names:
            names.append(name)
        return ""

    prompt = re.sub(
        r"人物\s*=\s*([^\s#@=，,]+)",
        lambda match: collect("characters", character_names, match.group(1)),
        prompt,
    )
    prompt = re.sub(
        r"风格\s*=\s*([^\s#@=，,]+)",
        lambda match: collect("styles", style_names, match.group(1)),
        prompt,
    )
    prompt = re.sub(
        r"@([^\s#@=，,]+)",
        lambda match: collect("characters", character_names, match.group(1)),
        prompt,
    )
    prompt = re.sub(
        r"#([^\s@=，,]+)",
        lambda match: collect("styles", style_names, match.group(1)),
        prompt,
    )

    all_presets = preset_store.all()
    _delimiters = set(" ,，、。！？\t\n")
    for kind, names_list in (("styles", style_names), ("characters", character_names)):
        known = sorted(all_presets[kind].keys(), key=len, reverse=True)
        for name in known:
            if name in names_list:
                continue
            idx = prompt.find(name)
            if idx < 0:
                continue
            before_ok = idx == 0 or prompt[idx - 1] in _delimiters
            after_idx = idx + len(name)
            after_ok = after_idx >= len(prompt) or prompt[after_idx] in _delimiters
            if before_ok and after_ok:
                names_list.append(name)
                prompt = prompt[:idx] + prompt[after_idx:]

    missing = []
    expanded = []
    for kind, names in (("characters", character_names), ("styles", style_names)):
        for name in names:
            value = preset_store.get(kind, name)
            if value is None:
                missing.append(f"{PRESET_KIND_LABELS[kind]}预设「{name}」")
            else:
                expanded.append(value)
    if missing:
        raise ValueError("未找到" + "、".join(missing))

    prompt = re.sub(r"\s+", " ", prompt).strip(" ,，")
    return ", ".join(expanded + ([prompt] if prompt else []))

NSFW_NEGATIVE_TAGS = "nsfw, nude, naked, nipples, pussy, penis, sex, vaginal, anal, oral, cum, ejaculation, pubic hair, genitals, exposed, uncensored"

TRANSLATE_SYSTEM_PROMPT = """You are a prompt translator for NovelAI image generation. Translate the user's description into English danbooru-style tags.
Rules:
1. Output ONLY comma-separated English tags, no explanations
2. Use danbooru tag format (lowercase, underscores for multi-word tags)
3. Always start with quality tags: masterpiece, best quality, very aesthetic
4. Include character count tags (1girl, 1boy, 2girls, etc.)
5. Translate clothing, pose, expression, background into appropriate tags
6. Keep any English tags the user already provided as-is
7. If input is already English tags, return with quality tags prepended
8. Do NOT add NSFW tags unless explicitly requested"""


def _get_tokens() -> list:
    raw = config.API_TOKEN or ""
    return [t.strip() for t in re.split(r"[,;\n]+", raw) if t.strip()]


def _get_current_token() -> str:
    global _token_index
    tokens = _get_tokens()
    if not tokens:
        raise ValueError("NovelAI Token 未配置，请在插件设置中填写 API_TOKEN。")
    _token_index = _token_index % len(tokens)
    return tokens[_token_index]


def _rotate_token(reason: str) -> bool:
    global _token_index
    tokens = _get_tokens()
    if len(tokens) <= 1:
        return False
    _token_index = (_token_index + 1) % len(tokens)
    logger.warning(f"NovelAI Token 轮切: {reason} -> 第 {_token_index + 1}/{len(tokens)} 个")
    return True


def _get_api_url(endpoint: str = "generate-image") -> str:
    if config.API_PROXY:
        return f"{config.API_PROXY.rstrip('/')}/ai/{endpoint}"
    return f"{NAI_API_BASE}/ai/{endpoint}"


def _parse_resolution() -> tuple:
    res = config.DEFAULT_RESOLUTION or "832x1216"
    m = re.match(r"(\d+)[xX×](\d+)", res)
    return (int(m.group(1)), int(m.group(2))) if m else (832, 1216)


def _build_client_config() -> dict:
    cfg: dict = {"timeout": httpx.Timeout(90.0, connect=30.0), "limits": httpx.Limits(max_keepalive_connections=2, max_connections=3)}
    if config.HTTP_PROXY:
        cfg["proxy"] = config.HTTP_PROXY
    return cfg


def _build_headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_current_token()}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Connection": "close",
    }


def _get_negative_prompt(extra: str = "") -> str:
    parts = []
    if config.NEGATIVE_PROMPT:
        parts.append(config.NEGATIVE_PROMPT)
    if not config.ENABLE_R18:
        parts.append(NSFW_NEGATIVE_TAGS)
    if extra:
        parts.append(extra)
    return ", ".join(parts)


def _build_parameters(prompt, width, height, steps, scale, sampler, model, negative_prompt=""):
    neg = _get_negative_prompt(negative_prompt)
    params = {
        "params_version": 3, "width": width, "height": height, "scale": scale,
        "sampler": sampler, "steps": steps, "n_samples": 1, "ucPreset": 0,
        "qualityToggle": True, "dynamic_thresholding": False, "controlnet_strength": 1,
        "legacy": False, "add_original_image": False, "cfg_rescale": config.CFG_RESCALE,
        "noise_schedule": "karras", "legacy_v3_extend": False, "skip_cfg_above_sigma": None,
        "use_coords": False, "characterPrompts": [], "negative_prompt": neg,
        "seed": random.randint(0, 4294967295),
    }
    if _is_v4_style(model):
        params["v4_prompt"] = _create_v4_prompt(prompt)
        params["v4_negative_prompt"] = {"caption": {"base_caption": neg, "char_captions": [], "scenery_captions": []}}
    return params


def _strip_png_metadata(data: bytes) -> bytes:
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(data))
        clean = PILImage.new(img.mode, img.size)
        clean.putdata(list(img.getdata()))
        buf = io.BytesIO()
        clean.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return data


def _extract_image(response_data: bytes) -> Optional[bytes]:
    try:
        from PIL import Image as PILImage
        try:
            img = PILImage.open(io.BytesIO(response_data))
            img.close()
            return response_data
        except Exception:
            pass
    except ImportError:
        if response_data[:4] == b"\x89PNG" or response_data[:2] == b"\xff\xd8":
            return response_data
    try:
        with zipfile.ZipFile(io.BytesIO(response_data)) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    return zf.read(name)
    except Exception:
        pass
    try:
        data = json.loads(response_data.decode("utf-8"))
        if "image" in data:
            return base64.b64decode(data["image"])
    except Exception:
        pass
    return response_data


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


async def _llm_translate_segment(client, chat_model: str, text: str) -> str:
    try:
        response = await client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": f"请将以下描述翻译为 NovelAI 画图提示词（danbooru 标签格式），不要添加 quality 标签：\n\n{text}"},
            ],
            max_tokens=300, temperature=0.3,
        )
        translated = response.choices[0].message.content.strip() if response.choices else ""
        return translated if translated else text
    except Exception:
        return text


async def _translate_prompt(prompt: str) -> str:
    if not config.TRANSLATE_MODEL_GROUP:
        return prompt
    if not _has_chinese(prompt):
        return prompt
    try:
        from openai import AsyncOpenAI
        from nekro_agent.core.config import config as global_config
        group_key = config.TRANSLATE_MODEL_GROUP
        if group_key not in global_config.MODEL_GROUPS:
            logger.warning(f"NovelAI 翻译: 模型组 \'{group_key}\' 未配置，跳过翻译")
            return prompt
        mg = global_config.MODEL_GROUPS[group_key]
        api_key = str(getattr(mg, "API_KEY", ""))
        base_url = str(getattr(mg, "BASE_URL", ""))
        chat_model = str(getattr(mg, "CHAT_MODEL", ""))
        if not api_key or not base_url:
            logger.warning("NovelAI 翻译: 模型组缺少 API_KEY 或 BASE_URL，跳过翻译")
            return prompt
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30)
        parts = [p.strip() for p in re.split(r"[,，]", prompt) if p.strip()]
        result_parts: list[str] = []
        to_translate: list[tuple[int, str]] = []
        for i, part in enumerate(parts):
            if _has_chinese(part):
                to_translate.append((i, part))
                result_parts.append("")
            else:
                result_parts.append(part)
        if not to_translate:
            return prompt
        translated = await asyncio.gather(
            *[_llm_translate_segment(client, chat_model, text) for _, text in to_translate]
        )
        for (idx, _orig), trans in zip(to_translate, translated):
            result_parts[idx] = trans
        final = ", ".join(p for p in result_parts if p)
        logger.info(f"NovelAI 翻译: {prompt[:50]}... -> {final[:80]}...")
        return final
    except ImportError:
        logger.warning("NovelAI 翻译: openai 库不可用，跳过翻译")
        return prompt
    except Exception as exc:
        logger.warning(f"NovelAI 翻译失败: {exc}，使用原始提示词")
        return prompt


async def _call_txt2img(prompt, width=0, height=0, steps=0, scale=0, sampler="", model="", negative_prompt=""):
    model = model or config.DEFAULT_MODEL
    dw, dh = _parse_resolution()
    width, height = width or dw, height or dh
    steps = steps or config.DEFAULT_STEPS
    scale = scale or config.DEFAULT_SCALE
    sampler = sampler or config.DEFAULT_SAMPLER
    prompt = await _translate_prompt(_expand_preset_prompt(prompt))
    params = _build_parameters(prompt, width, height, steps, scale, sampler, model, negative_prompt)
    payload = {"input": prompt, "model": model, "action": "generate", "parameters": params}
    api_url = _get_api_url("generate-image")
    max_retries = 3 if len(_get_tokens()) > 1 else 2
    async with httpx.AsyncClient(**_build_client_config()) as client:
        for retry in range(max_retries):
            try:
                if retry > 0:
                    await asyncio.sleep(retry * 1.0)
                headers = _build_headers()
                response = await client.post(api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    img = _extract_image(response.content)
                    if img is None:
                        raise Exception("无法从响应中提取图片")
                    logger.info(f"NovelAI 文生图成功: model={model} ({_model_label(model)}), {width}x{height}")
                    return img
                elif response.status_code in (401, 402, 429):
                    reason = f"HTTP {response.status_code}: {response.text[:60]}"
                    if retry < max_retries - 1 and _rotate_token(reason):
                        continue
                    raise Exception(f"NovelAI API 认证/配额错误: {reason}")
                elif response.status_code >= 500:
                    if retry < max_retries - 1:
                        logger.warning(f"NovelAI 服务器错误 {response.status_code}，将重试")
                        continue
                    raise Exception(f"NovelAI 服务器错误: {response.status_code}")
                else:
                    raise Exception(f"NovelAI API 错误: {response.status_code} {response.text[:200]}")
            except httpx.RequestError as exc:
                if retry < max_retries - 1:
                    logger.warning(f"网络请求失败: {exc}，将重试")
                    continue
                raise Exception(f"NovelAI 网络错误: {exc}") from exc
    raise Exception("NovelAI 请求失败: 超过最大重试次数")


async def _call_img2img(prompt, image_b64, width=0, height=0, steps=0, scale=0, strength=0, noise=-1, model="", negative_prompt=""):
    model = model or config.DEFAULT_MODEL
    dw, dh = _parse_resolution()
    width, height = width or dw, height or dh
    steps = steps or config.DEFAULT_STEPS
    scale = scale or config.DEFAULT_SCALE
    strength = strength or config.IMG2IMG_STRENGTH
    noise = noise if noise >= 0 else config.IMG2IMG_NOISE
    prompt = await _translate_prompt(_expand_preset_prompt(prompt))
    params = _build_parameters(prompt, width, height, steps, scale, config.DEFAULT_SAMPLER, model, negative_prompt)
    params["image"] = image_b64
    params["strength"] = strength
    params["noise"] = noise
    payload = {"input": prompt, "model": model, "action": "img2img", "parameters": params}
    api_url = _get_api_url("generate-image")
    async with httpx.AsyncClient(**_build_client_config()) as client:
        headers = _build_headers()
        response = await client.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            img = _extract_image(response.content)
            if img is None:
                raise Exception("无法从响应中提取图片")
            logger.info(f"NovelAI 图生图成功: model={model}, strength={strength}, noise={noise}")
            return img
        elif response.status_code in (401, 402, 429):
            reason = f"HTTP {response.status_code}: {response.text[:60]}"
            if _rotate_token(reason):
                headers = _build_headers()
                response = await client.post(api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    img = _extract_image(response.content)
                    if img is None:
                        raise Exception("无法从响应中提取图片")
                    return img
            raise Exception(f"NovelAI API 认证/配额错误: {reason}")
        else:
            raise Exception(f"NovelAI API 错误: {response.status_code} {response.text[:200]}")


async def _forward_result(ctx: AgentCtx, image_data: bytes, fmt: str = "png") -> str:
    shared_root = Path(ctx.fs.shared_path).resolve()
    shared_root.mkdir(parents=True, exist_ok=True)
    filename = f"novelai_{random.randint(100000, 999999)}.{fmt}"
    file_path = shared_root / filename
    file_path.write_bytes(_strip_png_metadata(image_data))
    send_path = ctx.fs.forward_file(file_path)
    if asyncio.iscoroutine(send_path) or asyncio.isfuture(send_path):
        send_path = await send_path
    return str(send_path)


RATIO_PRESETS = {
    "竖": (832, 1216), "portrait": (832, 1216), "2:3": (832, 1216),
    "方": (1024, 1024), "square": (1024, 1024), "1:1": (1024, 1024),
    "横": (1216, 832), "landscape": (1216, 832), "3:2": (1216, 832),
}


def _parse_size(size_str: str) -> tuple:
    if not size_str or size_str == "auto":
        return 0, 0
    s = size_str.strip().lower()
    if s in RATIO_PRESETS:
        return RATIO_PRESETS[s]
    m = re.match(r"(\d+)\s*[xX\u00d7]\s*(\d+)", size_str)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _extract_draw_params(prompt: str) -> tuple:
    """从 prompt 中提取尺寸关键词（竖/方/横），返回 (clean_prompt, width, height)。"""
    width, height = 0, 0
    _delims = set(" ,，、\t\n")
    for keyword in ("竖", "横", "方"):
        best = -1
        start = 0
        while True:
            idx = prompt.find(keyword, start)
            if idx < 0:
                break
            before_ok = idx == 0 or prompt[idx - 1] in _delims
            after_idx = idx + len(keyword)
            after_ok = after_idx >= len(prompt) or prompt[after_idx] in _delims
            if before_ok and after_ok:
                best = idx
                break
            start = idx + 1
        if best >= 0:
            width, height = _parse_size(keyword)
            prompt = prompt[:best] + prompt[best + len(keyword):]
            break
    return prompt.strip(" ,，"), width, height


@plugin.mount_command(
    name="nai",
    description="使用 NovelAI 生成图片",
    aliases=["画图", "nai画图", "novelai"],
    permission=CommandPermission.PUBLIC,
    usage="nai <提示词> [竖/方/横]",
)
async def cmd_draw(
    context: CommandExecutionContext,
    prompt: Annotated[str, Arg("画图提示词", positional=True, greedy=True)] = "",
) -> AsyncIterator[CommandResponse]:
    """使用 NovelAI 生成图片。支持中文描述（自动翻译）和英文 danbooru 标签。支持 -w宽 -h高 -r比例 或 WxH 尺寸参数。"""
    if not prompt.strip():
        yield CmdCtl.failed("请提供画图提示词，例如: /nai 一个穿白裙子的少女站在花田里\n加「竖」「方」「横」可指定尺寸")
        return

    clean_prompt, width, height = _extract_draw_params(prompt)
    yield CmdCtl.message("🎨 正在绘制中，请稍候...")

    try:
        image_data = await _call_txt2img(prompt=clean_prompt.strip(), width=width, height=height)
    except Exception as exc:
        yield CmdCtl.failed(f"NovelAI 画图失败: {exc}")
        return
    try:
        save_dir = plugin.get_plugin_data_dir() / "generated"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"nai_{int(_time.time())}.png"
        file_path = save_dir / file_name
        file_path.write_bytes(_strip_png_metadata(image_data))
        abs_path = str(file_path.resolve())
        chat_key = getattr(context, "chat_key", "") or ""
        _last_draw[chat_key] = {"prompt": prompt, "width": width, "height": height}
        yield CmdCtl.success([
            CommandOutputSegment(type=CommandOutputSegmentType.TEXT, text="NovelAI 画图完成"),
            CommandOutputSegment(type=CommandOutputSegmentType.IMAGE, file_path=abs_path),
        ])
    except Exception:
        yield CmdCtl.failed("图片保存失败")



@plugin.mount_command(
    name="重画",
    description="使用上一次的参数重新画图（可追加新描述）",
    aliases=["redraw"],
    permission=CommandPermission.PUBLIC,
    usage="重画 [追加描述]",
)
async def cmd_redraw(
    context: CommandExecutionContext,
    extra: Annotated[str, Arg("追加描述", positional=True, greedy=True)] = "",
) -> AsyncIterator[CommandResponse]:
    chat_key = getattr(context, "chat_key", "") or ""
    last = _last_draw.get(chat_key)
    if not last:
        yield CmdCtl.failed("还没有画过图，请先使用 /画图 命令。")
        return
    prompt = last["prompt"]
    if extra.strip():
        prompt = extra.strip() + "，" + prompt
    clean_prompt, width, height = _extract_draw_params(prompt)
    width = width or last.get("width", 0)
    height = height or last.get("height", 0)
    yield CmdCtl.message("🎨 正在重画，请稍候...")
    try:
        image_data = await _call_txt2img(prompt=clean_prompt.strip(), width=width, height=height)
    except Exception as exc:
        yield CmdCtl.failed(f"NovelAI 重画失败: {exc}")
        return
    try:
        save_dir = plugin.get_plugin_data_dir() / "generated"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"nai_{int(_time.time())}.png"
        file_path = save_dir / file_name
        file_path.write_bytes(_strip_png_metadata(image_data))
        abs_path = str(file_path.resolve())
        raw_path = save_dir / f"raw_{file_name}"
        raw_path.write_bytes(image_data)
        _last_draw[chat_key] = {"prompt": prompt, "width": width, "height": height, "image_path": str(raw_path.resolve())}
        yield CmdCtl.success([
            CommandOutputSegment(type=CommandOutputSegmentType.TEXT, text="NovelAI 重画完成"),
            CommandOutputSegment(type=CommandOutputSegmentType.IMAGE, file_path=abs_path),
        ])
    except Exception:
        yield CmdCtl.failed("图片保存失败")


def _preset_list_text(kind: str) -> str:
    values = preset_store.all()[kind]
    if not values:
        return f"当前没有{PRESET_KIND_LABELS[kind]}预设。"
    lines = [f"{PRESET_KIND_LABELS[kind]}预设："]
    lines.extend(f"- {name}: {prompt}" for name, prompt in values.items())
    return "\n".join(lines)


def _preset_command_help(kind: str) -> str:
    label = PRESET_KIND_LABELS[kind]
    return f"用法：/添加{label} 名称 提示词\n例如：/添加{label} 示例 masterpiece, best quality"


@plugin.mount_command(
    name="添加人物",
    description="添加或修改一个人物预设",
    permission=CommandPermission.ADVANCED,
    usage="添加人物 <名称> <提示词>",
)
async def cmd_add_character(
    context: CommandExecutionContext,
    name: Annotated[str, Arg("人物名称", positional=True)] = "",
    prompt: Annotated[str, Arg("人物提示词", positional=True, greedy=True)] = "",
) -> AsyncIterator[CommandResponse]:
    if not name.strip() or not prompt.strip():
        yield CmdCtl.failed(_preset_command_help("characters"))
        return
    try:
        preset_store.set("characters", name, prompt)
        yield CmdCtl.message(f"人物预设「{name.strip()}」已保存。")
    except ValueError as exc:
        yield CmdCtl.failed(str(exc))


@plugin.mount_command(
    name="删除人物",
    description="删除一个人物预设",
    permission=CommandPermission.ADVANCED,
    usage="删除人物 <名称>",
)
async def cmd_delete_character(
    context: CommandExecutionContext,
    name: Annotated[str, Arg("人物名称", positional=True)] = "",
) -> AsyncIterator[CommandResponse]:
    if not name.strip():
        yield CmdCtl.failed("用法：/删除人物 名称")
        return
    if preset_store.delete("characters", name):
        yield CmdCtl.message(f"人物预设「{name.strip()}」已删除。")
    else:
        yield CmdCtl.failed(f"未找到人物预设「{name.strip()}」。")


@plugin.mount_command(
    name="人物列表",
    description="查看人物预设列表",
    permission=CommandPermission.PUBLIC,
    usage="人物列表",
)
async def cmd_list_characters(context: CommandExecutionContext) -> AsyncIterator[CommandResponse]:
    yield CmdCtl.message(_preset_list_text("characters"))


@plugin.mount_command(
    name="添加风格",
    description="添加或修改一个风格预设",
    permission=CommandPermission.ADVANCED,
    usage="添加风格 <名称> <提示词>",
)
async def cmd_add_style(
    context: CommandExecutionContext,
    name: Annotated[str, Arg("风格名称", positional=True)] = "",
    prompt: Annotated[str, Arg("风格提示词", positional=True, greedy=True)] = "",
) -> AsyncIterator[CommandResponse]:
    if not name.strip() or not prompt.strip():
        yield CmdCtl.failed(_preset_command_help("styles"))
        return
    try:
        preset_store.set("styles", name, prompt)
        yield CmdCtl.message(f"风格预设「{name.strip()}」已保存。")
    except ValueError as exc:
        yield CmdCtl.failed(str(exc))


@plugin.mount_command(
    name="删除风格",
    description="删除一个风格预设",
    permission=CommandPermission.ADVANCED,
    usage="删除风格 <名称>",
)
async def cmd_delete_style(
    context: CommandExecutionContext,
    name: Annotated[str, Arg("风格名称", positional=True)] = "",
) -> AsyncIterator[CommandResponse]:
    if not name.strip():
        yield CmdCtl.failed("用法：/删除风格 名称")
        return
    if preset_store.delete("styles", name):
        yield CmdCtl.message(f"风格预设「{name.strip()}」已删除。")
    else:
        yield CmdCtl.failed(f"未找到风格预设「{name.strip()}」。")


@plugin.mount_command(
    name="风格列表",
    description="查看风格预设列表",
    permission=CommandPermission.PUBLIC,
    usage="风格列表",
)
async def cmd_list_styles(context: CommandExecutionContext) -> AsyncIterator[CommandResponse]:
    yield CmdCtl.message(_preset_list_text("styles"))



@plugin.mount_command(
    name="看参数",
    description="提取 NovelAI 图片的生成参数（PNG 元数据）",
    aliases=["反推", "查看参数", "naimeta"],
    permission=CommandPermission.PUBLIC,
    usage="看参数 (回复一张图片)",
)
async def cmd_metadata(
    context: CommandExecutionContext,
    prompt: Annotated[str, Arg("提示", positional=True, greedy=True)] = "",
) -> AsyncIterator[CommandResponse]:
    image_url = None
    if hasattr(context, "event") and context.event:
        event = context.event
        for seg in getattr(event, "message", []):
            seg_data = seg if isinstance(seg, dict) else (seg.data if hasattr(seg, "data") else {})
            seg_type = seg.get("type", "") if isinstance(seg, dict) else getattr(seg, "type", "")
            if seg_type == "image":
                image_url = seg_data.get("url") or seg_data.get("file")
                break
            if seg_type == "reply":
                reply_msg = getattr(event, "reply", None)
                if reply_msg and hasattr(reply_msg, "message"):
                    for rseg in reply_msg.message:
                        rd = rseg if isinstance(rseg, dict) else (rseg.data if hasattr(rseg, "data") else {})
                        rt = rseg.get("type", "") if isinstance(rseg, dict) else getattr(rseg, "type", "")
                        if rt == "image":
                            image_url = rd.get("url") or rd.get("file")
                            break
    if not image_url:
        yield CmdCtl.failed("请回复一张图片来提取参数。")
        return
    yield CmdCtl.message("正在提取图片参数...")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            img_bytes = resp.content
    except Exception as exc:
        yield CmdCtl.failed(f"下载图片失败: {exc}")
        return
    metadata = _extract_png_metadata(img_bytes)
    if not metadata:
        yield CmdCtl.failed("未能从该图片中提取到 NovelAI 元数据。可能不是 NAI 生成的图片。")
        return
    yield CmdCtl.message(metadata)


def _extract_png_metadata(data: bytes) -> str:
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(data))
        info = {}
        if hasattr(img, "text"):
            info.update(img.text)
        for k, v in img.info.items():
            if isinstance(v, str) and k not in info:
                info[k] = v
        lines = []
        if "Description" in info:
            lines.append(f"📝 正面提示词:\n{info['Description']}")
        if "Comment" in info:
            try:
                comment = json.loads(info["Comment"])
                if "uc" in comment:
                    lines.append(f"🚫 负面提示词:\n{comment['uc']}")
                params = []
                for key in ("steps", "scale", "sampler", "seed", "width", "height", "noise_schedule", "cfg_rescale", "sm", "sm_dyn", "strength", "noise"):
                    if key in comment:
                        params.append(f"{key}: {comment[key]}")
                if params:
                    lines.append("⚙️ 参数: " + " | ".join(params))
            except (json.JSONDecodeError, TypeError):
                lines.append(f"💬 Comment: {info['Comment'][:500]}")
        if "Software" in info:
            lines.append(f"🔧 Software: {info['Software']}")
        if "Source" in info:
            lines.append(f"🏷️ Model: {info['Source']}")
        if not lines and info:
            for k, v in info.items():
                if isinstance(v, str) and len(v) < 2000:
                    lines.append(f"{k}: {v[:500]}")
        if not lines:
            stealth = _try_stealth_pnginfo(data)
            if stealth:
                lines.append(f"📝 隐写元数据:\n{stealth[:2000]}")
        if not lines:
            return ""
        return "\n\n".join(lines)
    except ImportError:
        return ""
    except Exception:
        return ""


def _try_stealth_pnginfo(data: bytes) -> str:
    try:
        import gzip
        from PIL import Image as PILImage
        import numpy as np
        img = PILImage.open(io.BytesIO(data))
        if img.mode != "RGBA":
            return ""
        pixels = np.array(img)
        alpha = pixels[:, :, 3].flatten()
        bits = alpha & 1
        byte_count = len(bits) // 8
        raw_bytes = np.packbits(bits[:byte_count * 8])
        raw = bytes(raw_bytes)
        for magic in (b"stealth_pnginfo", b"stealth_pngcomp"):
            idx = raw.find(magic)
            if idx >= 0:
                payload = raw[idx + len(magic):]
                if magic == b"stealth_pngcomp":
                    try:
                        payload = gzip.decompress(payload)
                    except Exception:
                        pass
                text = payload.decode("utf-8", errors="ignore")
                end = text.find("\x00")
                return text[:end] if end > 0 else text[:2000]
        return ""
    except Exception:
        return ""


PRESET_WEBUI_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NovelAI 预设管理</title>
<style>
:root { color-scheme: dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; padding: 24px; background: #111827; color: #e5e7eb; }
main { max-width: 980px; margin: 0 auto; }
h1 { margin: 0 0 8px; font-size: 28px; }
.hint { color: #9ca3af; margin: 0 0 22px; }
.tabs { display: flex; gap: 8px; margin-bottom: 16px; }
button { border: 0; border-radius: 8px; padding: 9px 15px; cursor: pointer; color: #fff; background: #374151; font-size: 14px; }
button.active, button.primary { background: #7c3aed; }
button.danger { background: #b91c1c; }
.panel { background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 18px; }
label { display: block; margin: 10px 0 6px; color: #d1d5db; }
input, textarea { box-sizing: border-box; width: 100%; border: 1px solid #4b5563; border-radius: 8px; padding: 10px; background: #111827; color: #f9fafb; font: inherit; }
textarea { min-height: 100px; resize: vertical; }
.actions { display: flex; gap: 8px; margin-top: 14px; }
.items { display: grid; gap: 10px; margin-top: 18px; }
.item { border: 1px solid #374151; border-radius: 9px; padding: 12px; background: #182230; }
.item-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.name { font-weight: 700; color: #c4b5fd; }
.prompt { margin: 8px 0 0; color: #d1d5db; white-space: pre-wrap; word-break: break-word; }
.item-actions { display: flex; gap: 6px; }
#status { min-height: 22px; margin-top: 12px; color: #a7f3d0; }
.empty { color: #9ca3af; padding: 10px 0; }
</style>
</head>
<body>
<main>
<h1>NovelAI 预设管理</h1>
<p class="hint">保存后可在画图指令中引用，直接写名称即可（也支持 <code>@人物名</code>、<code>#风格名</code>），例如：<code>/画图 风堇 风格052 花田</code></p>
<div class="tabs"><button id="characters-tab" class="active" onclick="switchKind('characters')">人物预设</button><button id="styles-tab" onclick="switchKind('styles')">风格预设</button></div>
<section class="panel">
<form id="preset-form" onsubmit="savePreset(event)">
<label for="preset-name">名称</label>
<input id="preset-name" maxlength="40" required placeholder="例如：风堇">
<label for="preset-prompt">提示词</label>
<textarea id="preset-prompt" maxlength="4000" required placeholder="例如：hyacine, honkai star rail, 1girl"></textarea>
<div class="actions"><button class="primary" type="submit">保存预设</button><button type="button" onclick="clearForm()">清空</button></div>
</form>
<div id="status"></div>
<div id="items" class="items"></div>
</section>
</main>
<script>
let currentKind = 'characters';
let presets = { characters: {}, styles: {} };
const labels = { characters: '人物', styles: '风格' };
const el = (id) => document.getElementById(id);
function setStatus(message, error = false) { const node = el('status'); node.textContent = message; node.style.color = error ? '#fca5a5' : '#a7f3d0'; }
function switchKind(kind) { currentKind = kind; document.querySelectorAll('.tabs button').forEach((node) => node.classList.remove('active')); el(kind + '-tab').classList.add('active'); clearForm(); render(); }
function clearForm() { el('preset-name').value = ''; el('preset-prompt').value = ''; }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char])); }
function render() {
  const node = el('items');
  const entries = Object.entries(presets[currentKind] || {});
  if (!entries.length) { node.innerHTML = '<div class="empty">还没有' + labels[currentKind] + '预设。</div>'; return; }
  node.innerHTML = entries.map(([name, prompt]) => '<article class="item"><div class="item-head"><span class="name">' + escapeHtml(name) + '</span><span class="item-actions"><button type="button" onclick="editPreset(' + JSON.stringify(name) + ')">编辑</button><button class="danger" type="button" onclick="removePreset(' + JSON.stringify(name) + ')">删除</button></span></div><div class="prompt">' + escapeHtml(prompt) + '</div></article>').join('');
}
async function loadPresets() {
  try { const response = await fetch('api/presets'); if (!response.ok) throw new Error('加载失败'); presets = await response.json(); render(); }
  catch (error) { setStatus(error.message, true); }
}
function editPreset(name) { el('preset-name').value = name; el('preset-prompt').value = presets[currentKind][name] || ''; el('preset-name').focus(); }
async function savePreset(event) {
  event.preventDefault();
  try {
    const response = await fetch('api/presets/' + currentKind, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: el('preset-name').value, prompt: el('preset-prompt').value }) });
    const result = await response.json(); if (!response.ok) throw new Error(result.detail || '保存失败');
    presets = result; render(); setStatus('保存成功'); clearForm();
  } catch (error) { setStatus(error.message, true); }
}
async function removePreset(name) {
  if (!confirm('确定删除「' + name + '」吗？')) return;
  try { const response = await fetch('api/presets/' + currentKind + '/' + encodeURIComponent(name), { method: 'DELETE' }); const result = await response.json(); if (!response.ok) throw new Error(result.detail || '删除失败'); presets = result; render(); setStatus('删除成功'); }
  catch (error) { setStatus(error.message, true); }
}
loadPresets();
</script>
</body>
</html>"""


@plugin.mount_router()
def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def preset_manager_page() -> HTMLResponse:
        return HTMLResponse(content=PRESET_WEBUI_HTML)

    @router.get("/api/presets")
    async def api_get_presets() -> dict:
        return preset_store.all()

    @router.post("/api/presets/{kind}")
    async def api_save_preset(kind: str, payload: dict = Body(...)) -> dict:
        try:
            preset_store.set(kind, str(payload.get("name", "")), str(payload.get("prompt", "")))
            return preset_store.all()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/api/presets/{kind}/{name}")
    async def api_delete_preset(kind: str, name: str) -> dict:
        try:
            if not preset_store.delete(kind, name):
                raise HTTPException(status_code=404, detail=f"未找到{PRESET_KIND_LABELS[kind]}预设「{name}」")
            return preset_store.all()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="NovelAI 文生图",
    description="使用 NovelAI 模型生成高质量动漫/插画图片",
)
async def novelai_generate(
    _ctx: AgentCtx,
    prompt: str,
    size: str = "",
    negative_prompt: str = "",
    model: str = "",
    send_to_chat: bool = True,
) -> str:
    """Generate an image with NovelAI.

    Args:
        prompt: Image description. Can be Chinese (auto-translated) or English danbooru tags.
        size: Optional size as WIDTHxHEIGHT. Empty uses plugin default.
        negative_prompt: Additional negative prompt tags to avoid.
        model: Model name override. Empty uses plugin default.
        send_to_chat: Send the generated image to chat.

    Returns:
        The generated image sandbox path.
    """
    width, height = _parse_size(size)
    image_data = await _call_txt2img(prompt=prompt, width=width, height=height, model=model, negative_prompt=negative_prompt)
    path = await _forward_result(_ctx, image_data)
    if send_to_chat:
        await _ctx.send_image(path)
    return path


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="NovelAI 图生图",
    description="使用 NovelAI 模型以参考图片生成新图片（图生图）",
)
async def novelai_img2img(
    _ctx: AgentCtx,
    image_path: str,
    prompt: str,
    size: str = "",
    strength: float = 0,
    noise: float = -1,
    negative_prompt: str = "",
    model: str = "",
    send_to_chat: bool = True,
) -> str:
    """Generate an image based on a reference image with NovelAI (img2img).

    Args:
        image_path: Sandbox path to the reference image.
        prompt: Prompt describing the desired output. Can be Chinese or English tags.
        size: Optional size as WIDTHxHEIGHT. Empty uses plugin default.
        strength: img2img strength (0.1-0.9). 0 uses plugin default.
        noise: img2img noise (0.0-1.0). -1 uses plugin default.
        negative_prompt: Additional negative prompt tags.
        model: Model name override. Empty uses plugin default.
        send_to_chat: Send the result to chat.

    Returns:
        The generated image sandbox path.
    """
    ref_path = Path(image_path)
    if not ref_path.exists():
        sandbox_path = Path(_ctx.fs.sandbox_path) / image_path.lstrip("/")
        if sandbox_path.exists():
            ref_path = sandbox_path
        else:
            raise FileNotFoundError(f"参考图片不存在: {image_path}")
    image_bytes = ref_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    width, height = _parse_size(size)
    image_data = await _call_img2img(
        prompt=prompt, image_b64=image_b64, width=width, height=height,
        strength=strength, noise=noise, model=model, negative_prompt=negative_prompt,
    )
    path = await _forward_result(_ctx, image_data)
    if send_to_chat:
        await _ctx.send_image(path)
    return path
