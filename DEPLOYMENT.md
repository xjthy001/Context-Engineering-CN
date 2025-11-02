# 部署说明

本项目使用 VitePress 构建静态网站，可以部署到多个平台。

## 🚀 部署选项

### 选项 1: GitHub Pages (推荐)

#### 自动部署

已配置 GitHub Actions 自动部署工作流 (`.github/workflows/deploy.yml`)。

**步骤**:
1. 在 GitHub 仓库设置中启用 GitHub Pages
   - 进入 `Settings` → `Pages`
   - Source 选择: `GitHub Actions`

2. 推送代码到 main 分支即可自动触发部署

3. 访问: `https://xjthy001.github.io/Context-Engineering-CN/`

#### 手动部署

```bash
# 构建
npm run docs:build

# 部署到 GitHub Pages
# 需要安装 gh-pages
npm install -g gh-pages
gh-pages -d .vitepress/dist
```

---

### 选项 2: Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/xjthy001/Context-Engineering-CN)

**步骤**:
1. 导入 GitHub 仓库到 Vercel
2. 配置构建设置:
   - **Build Command**: `npm run docs:build`
   - **Output Directory**: `.vitepress/dist`
   - **Install Command**: `npm install`

3. 点击 Deploy

**自定义域名**:
- 在 Vercel 项目设置中添加自定义域名
- 更新 `.vitepress/config.mts` 中的 `base` 配置为 `/`

---

### 选项 3: Cloudflare Pages

**步骤**:
1. 登录 Cloudflare Dashboard
2. 进入 Pages → Create a project
3. 连接 GitHub 仓库
4. 配置构建设置:
   - **Build command**: `npm run docs:build`
   - **Build output directory**: `.vitepress/dist`
   - **Root directory**: `/`

5. 点击 Save and Deploy

**环境变量**:
- `NODE_VERSION`: `20`

---

### 选项 4: Netlify

**步骤**:
1. 登录 Netlify
2. 点击 `New site from Git`
3. 选择 GitHub 仓库
4. 配置构建设置:
   - **Build command**: `npm run docs:build`
   - **Publish directory**: `.vitepress/dist`

5. 点击 Deploy site

**netlify.toml** (可选):
```toml
[build]
  command = "npm run docs:build"
  publish = ".vitepress/dist"

[build.environment]
  NODE_VERSION = "20"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

---

## 🛠️ 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run docs:dev

# 构建生产版本
npm run docs:build

# 预览构建结果
npm run docs:preview
```

---

## ⚙️ 配置说明

### 基础路径配置

根据部署平台调整 `.vitepress/config.mts` 中的 `base` 配置:

```ts
// GitHub Pages 子路径
base: '/Context-Engineering-CN/'

// 根域名或自定义域名
base: '/'
```

### 主题配置

在 `.vitepress/config.mts` 中自定义:
- 导航栏 (`nav`)
- 侧边栏 (`sidebar`)
- 社交链接 (`socialLinks`)
- 搜索配置 (`search`)
- 页脚 (`footer`)

---

## 📊 构建统计

- **构建时间**: ~2-3分钟
- **输出大小**: ~50MB
- **页面数量**: 123+ 页面
- **资源文件**: 127+ 代码/配置文件

---

## 🐛 常见问题

### 1. 构建失败: `markdown-it-mathjax3` 错误

**解决**:
```bash
npm install -D markdown-it-mathjax3
```

### 2. 404 错误

**原因**: `base` 路径配置不正确

**解决**:
- GitHub Pages 子路径: `base: '/仓库名/'`
- 根域名: `base: '/'`

### 3. 样式不加载

**原因**: 资源路径错误

**解决**: 检查 `base` 配置和资源引用路径

### 4. 中文搜索不工作

**解决**: 确保 `search` 配置中包含中文 locale 设置

---

## 🔄 更新部署

### 自动更新 (GitHub Actions)

推送到 main 分支自动触发部署:
```bash
git add .
git commit -m "更新内容"
git push origin main
```

### 手动更新

1. 拉取最新代码
2. 重新构建
3. 部署到平台

---

## 📝 性能优化

### 1. 图片优化

使用 WebP 格式和适当的尺寸:
```bash
# 安装 sharp
npm install -D sharp

# 在构建脚本中添加图片优化
```

### 2. 代码分割

VitePress 自动进行代码分割，无需额外配置。

### 3. CDN 加速

部署到 Vercel/Cloudflare 自动获得全球 CDN 加速。

---

## 🌐 自定义域名

### GitHub Pages

1. 在仓库根目录添加 `CNAME` 文件
2. 文件内容: `your-domain.com`
3. 在域名 DNS 设置中添加 CNAME 记录

### Vercel/Cloudflare

在平台控制面板中添加自定义域名，按照提示配置 DNS。

---

## 📞 支持

遇到问题？
- [GitHub Issues](https://github.com/xjthy001/Context-Engineering-CN/issues)
- [VitePress 文档](https://vitepress.dev/)

---

**祝部署顺利！🎉**
