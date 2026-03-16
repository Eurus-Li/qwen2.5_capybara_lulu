水豚噜噜训练数据集获取说明

前提：你已经获得了噜噜素材的训练授权，并且只会采集授权范围内的页面或图片。

推荐获取顺序：

1. 权利方提供的原始图片包
2. 权利方提供的公开页面链接清单
3. 你获准访问的图床、文章页、活动页、素材页

不建议把抓取范围放大到未授权的第三方搬运站、转载页或账号页。

目录建议：

```text
data/
  raw/
  cleaned/
  captions/
manifests/
  lulu_urls.txt
dataset_tools/
  download_images.py
  lulu_config.example.json
```

脚本能力：

- 输入一个文本清单，逐条下载页面中的图片
- 输入一个 JSON 配置，递归抓取授权域名里的站内页面
- 识别 `og:image`、`twitter:image` 和普通 `img`
- 自动去重
- 自动写出 `metadata.csv`
- 可按最小尺寸过滤
- 可按域名白名单限制
- 可按图片 URL 正则过滤

最简单的用法：

```powershell
python .\dataset_tools\download_images.py `
  --input .\manifests\lulu_urls.txt `
  --output .\data\raw `
  --min-size 256 `
  --delay 1.0
```

如果你拿到的是一批授权站点，而不是单张图片链接，更适合用配置文件：

```powershell
python .\dataset_tools\download_images.py `
  --config .\dataset_tools\lulu_config.example.json `
  --output .\data\raw `
  --min-size 256 `
  --delay 1.0
```

配置项说明：

- `seeds`：起始页面
- `allow_domains`：允许抓取的域名白名单
- `max_depth`：站内递归深度
- `same_domain_only`：是否限制在当前域名
- `image_regex`：只保留匹配关键字的图片 URL

`lulu_urls.txt` 例子：

```text
https://example.com/lulu/post-1
https://example.com/lulu/post-2
https://example.com/assets/lulu-bath.jpg
```

`metadata.csv` 会记录：

- 本地保存路径
- 来源页面
- 图片 URL
- 图片哈希
- 宽高

建议第一版目标量：

- 原始图：100 到 300 张
- 清洗后可训练图：40 到 120 张

数据筛选标准：

- 保留：主体清楚、表情明显、角度不同、动作不同
- 删除：模糊、过暗、强遮挡、多人或多动物混杂、连续近乎重复的图

如果授权来源来自登录后页面或前端动态接口，这个通用脚本不一定能直接抓到。我可以继续帮你补一版更贴近具体站点的采集器，但前提仍然是你提供明确的授权来源范围。
