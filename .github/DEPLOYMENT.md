# 自动发布到 PerfectPixelArt.github.io

## 两个仓库的分工

```text
Bocchi-The-Glock/Perfect-PixelArt 的 main 分支收到 push
  → 安装 Python 3.12 和固定版本依赖
  → 运行 src/tests/test_pixelperfect.py
  → web/build.py 同步最新 Python 核心
  → 校验算法包、Pyodide、NumPy / Pillow 及许可证
  → 将 build/pages 的内容推送到目标仓库 gh-pages 分支
  → 目标仓库的 GitHub Pages 发布 https://perfectpixelart.github.io/
```

源仓库保留完整项目：`.github/`、`src/`、`input/`、`web/`、`pixelperfect.py`、`pyproject.toml` 等。不要只上传 web；测试需要 input 中的 6 张图片，构建需要 input/lastTour.png 和 src/pixelperfect。

目标仓库 `PerfectPixelArt/PerfectPixelArt.github.io` 接收网页成品。发布目录根部就是 index.html，同时包含前端 JS/CSS、core.zip、core-manifest.json、assets、vendor 和 .nojekyll。不会把开发脚本、测试或 input/output 文件夹复制到网站。

## 一、创建跨仓库令牌

使用对目标仓库有写权限、且属于 PerfectPixelArt 组织的账号：

1. GitHub 个人头像 → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**。
2. 名称可填 `Perfect-PixelArt website deploy`；设置有效期，到期前更新 Secret。
3. **Resource owner：PerfectPixelArt**。不要选源码仓库的个人用户名。
4. **Repository access → Only select repositories**：仅选择 `PerfectPixelArt.github.io`。
5. **Repository permissions → Contents：Read and write**；Metadata 的只读权限自动带上。
6. 生成令牌。如果显示 Pending，先让组织管理员批准。此方案只推送静态文件，不需要给令牌增加 Workflows、Pages 或组织管理权限。

如果 Resource owner 列表里没有 PerfectPixelArt，请先确认当前账号是该组织成员，以及组织允许 fine-grained tokens。若 PerfectPixelArt 实际是另一个个人账号，需要由该账号创建可写目标仓库的令牌。

## 二、Secret 放到源码仓库

打开 [源码仓库的 Actions secrets](https://github.com/Bocchi-The-Glock/Perfect-PixelArt/settings/secrets/actions)：

**Settings → Secrets and variables → Actions → New repository secret**

- Name：`ORG_PAGES_TOKEN`
- Secret：上一节生成的令牌值。

令牌的授权对象是**目标仓库**，保存 Secret 的位置是**源码仓库**。不要把令牌写进 YAML、提交记录或聊天。

源码仓库的 **Settings → Actions → General** 需要允许 workflow 使用 `actions/checkout`、`actions/setup-python` 和 `peaceiris/actions-gh-pages`。当前 workflow 的内置 GITHUB_TOKEN 只需 `contents: read`；跨仓库写入使用 ORG_PAGES_TOKEN。

## 三、首次推送，先生成 gh-pages 分支

将 `.github/workflows/deploy-org-pages.yml`、`.github/requirements-ci.txt`、`.github/scripts/prepare_pages.py` 和本说明提交到**源码仓库的 main 分支**。已有网页或 Python 修改也按你的发布需要提交。

注意 workflow 必须位于 GitHub 仓库根部的 `.github/workflows/`；不能外面再套一层 perfect_pixel_plus 文件夹。

进入 [源码仓库 Actions](https://github.com/Bocchi-The-Glock/Perfect-PixelArt/actions)，等待 **Deploy organization website** 成功。也可点进该 workflow → **Run workflow → Branch: main → Run workflow** 手动运行。

第一次成功后，目标仓库会出现 **gh-pages** 分支。不用预先创建；现有 main 分支不会被这个 workflow 改写。gh-pages 专门保存生成的网页：每次同步会移除旧的成品文件，保留提交历史，因此不要手动在 gh-pages 放其他资料。

## 四、在目标仓库开启 GitHub Pages（只需一次）

打开 [目标仓库 Pages 设置](https://github.com/PerfectPixelArt/PerfectPixelArt.github.io/settings/pages)：

1. **Settings → Pages → Build and deployment**。
2. **Source：Deploy from a branch**。
3. **Branch：gh-pages**，文件夹选 **/ (root)**，点 **Save**。
4. 等待目标仓库 Actions 中的 **pages build and deployment** 成功。
5. 访问 **https://perfectpixelart.github.io/**，用上传图片、生成、下载验证页面。

这里选择 Deploy from a branch；源码仓库的 Actions 负责构建和跨仓库推送，目标仓库负责从 gh-pages 发布。无需在目标仓库再复制这份源 workflow，也无需在源码仓库开启 Pages。

目标仓库目前若为空，Pages 下拉菜单可能没有 gh-pages；先完成第三节，再回来选择。首次开启 Pages 后，以目标仓库的部署记录和网站实际访问为准，源码 workflow 绿色只代表构建与推送成功。

## 之后怎样更新

每次向源码仓库 **main** 推送，或将 PR 合并进 main，都会运行检查并更新网页。其他分支的 push 不发布；手动运行也应选择 main。快速连续推送会串行处理，GitHub 可能合并等待中的旧任务，最终发布最新一次更新。

修改 Python 后不用手工更新算法压缩包再上传目标仓库：CI 会重新运行 web/build.py。vendor 已随源仓库提交，CI 校验现有运行时，不依赖每次从 CDN 重新下载。若更新运行时版本，应在开发环境重新执行 `python web/build.py --runtime`，检查清单与许可证后一起提交 vendor。

构建或测试失败时，不执行推送，已部署网站仍保留前一次版本。相同的网页成品不会强行创建空提交。

## 本地复现构建

在项目根目录、Python 3.12 环境运行：

```powershell
python -m pip install -r .github/requirements-ci.txt
python -m pytest src/tests/test_pixelperfect.py -q
python web/build.py
python web/build.py --check
python .github/scripts/prepare_pages.py
python -m http.server 8765 --bind 127.0.0.1 --directory build/pages
```

`build/pages` 是可重复生成的成品目录，每次 prepare_pages.py 都会替换它；build 已被 gitignore 排除。CI 执行 Python 回归及资源完整性检查，浏览器交互回归仍按 web/README.md 单独运行。

## 常见失败位置

| 现象 | 检查项 |
| --- | --- |
| Actions 中没有 workflow | 是否将 `.github` 提交到源仓库 main，文件是否在仓库根部，Actions 是否启用 |
| 提示缺少 ORG_PAGES_TOKEN | Secret 必须放源码仓库；名称完全一致；不要建成 Variable |
| 推送目标仓库报 403 / permission denied | 令牌是否过期；Resource owner/仓库选错；Contents 是否 Read and write；组织是否待审批；gh-pages 的 Rulesets 是否阻止直接推送 |
| Pytest 失败 | 读取失败项日志；先修复再发布，不跳过测试掩盖失败 |
| Missing / checksum mismatch | 检查 vendor 是否完整提交；排查 Git LFS 指针文件和误改；恢复对应文件后重新构建 |
| 源仓库 Actions 成功，网站仍 404 | 目标仓库 Pages 是否选 gh-pages / root，目标仓库部署是否成功 |
| 主页能开但一直加载算法 | 浏览器 Network 检查 core.zip、core-manifest.json、vendor/pyodide/*.wasm 和 wheel 是否 200；强制刷新 |
| 页面仍是旧版 | 确认修改已推送到源仓库 main，查看两个仓库最新部署，再 Ctrl+F5 |

## 参考

- [GitHub：设置 Pages 的发布来源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
- [GitHub：创建 fine-grained personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
- [actions-gh-pages：发布到另一仓库，需要 deploy key 或 personal token](https://github.com/peaceiris/actions-gh-pages#%EF%B8%8F-deploy-to-external-repository-external_repository)
