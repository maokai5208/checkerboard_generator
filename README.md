# 棋盘格生成工具

一个用于生成棋盘格标定板的Python工具，支持嵌入带坐标信息的Data Matrix码（DM码），并关联物理坐标系的原点与轴方向。

## 功能特性

- ✅ 可自定义棋盘格参数（行数、列数、方格尺寸）
- ✅ 生成标准棋盘格图案（第一个方格为黑色）
- ✅ 嵌入Data Matrix码（DM码），包含坐标信息
- ✅ 每个DM码占据2x2个方格，自动清除背景
- ✅ 支持按间距自动生成DM码位置
- ✅ 自动边界检查，超出边界的DM码会被跳过
- ✅ DM码边框自适应（1个模块大小）
- ✅ 自动保存配置文件（JSON格式）

## 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install numpy opencv-python Pillow pylibdmtx
```

## 快速开始

### 最简单的用法（推荐）

只需输入4个参数即可生成棋盘格：

```python
from checkerboard_generator import generate_checkerboard_simple

# 生成棋盘格
generate_checkerboard_simple(
    square_size=20.0,   # 棋盘格大小：20mm
    rows=25,          # 25行
    cols=25,           # 25列
    dm_spacing=5,      # DM码间距：每隔5个方格放置一个DM码
    dm_start_position=4  # 可选：第一个DM码从第5行第5列开始（如果不指定则使用dm_spacing）
)
```

### 直接运行

```bash
python checkerboard_generator.py
```

默认会生成一个25x25的棋盘格，每个方格20mm，DM码间距为5。

## 参数说明

### generate_checkerboard_simple() 参数

- `square_size` (float): 棋盘格大小（每个方格的物理尺寸，单位：mm）
- `rows` (int): 行数
- `cols` (int): 列数
- `dm_spacing` (int): DM码间距（每隔多少个方格放置一个DM码）
- `dm_start_position` (int, 可选): 第一个DM码的起始位置（行和列索引，从0开始）
  - 如果为None，则使用`dm_spacing`作为起始位置
  - 例如：如果设置为4，则第一个DM码从第5行第5列开始（从1开始计数）
- `output_path` (str, 可选): 输出文件路径（默认：`output/checkerboard_with_dm.png`）
  - 如果未指定，文件将自动保存到 `output/` 文件夹
  - 如果指定了路径但不包含 `output/` 目录，文件会自动保存到 `output/` 文件夹中

### 高级用法

如果需要更精细的控制，可以使用 `CheckerboardGenerator` 类：

```python
from checkerboard_generator import CheckerboardGenerator

# 创建棋盘格生成器
generator = CheckerboardGenerator(
    rows=25,
    cols=25,
    square_size=20.0,
    image_size=(2000, 2000),  # 图像尺寸
    margin=100                # 边距
)

# 生成并保存
generator.save_checkerboard(
    output_path="checkerboard.png",
    embed_dm=True,
    dm_spacing=5,
    dm_start_position=4,
    dm_square_size=2  # DM码占据2x2个方格
)
```

## DM码特性

- **尺寸**: 每个DM码占据2x2个方格（4个方格）
- **背景**: 自动清除DM码区域的棋盘格背景，填充白色
- **边框**: 自适应边框，大小为1个DM码内部模块大小
- **边界检查**: 自动检查边界，超出棋盘格区域的DM码会被跳过
- **坐标信息**: 每个DM码包含格式为 `行,列,G间距S棋盘格大小` 的坐标信息

### DM码数据格式

嵌入的Data Matrix码包含坐标信息，格式为：
```
行,列,G间距S棋盘格大小
```

例如：
```
5,5,G5S20.0
10,10,G5S20.0
```

其中：
- **行**: DM码所在的行号（从1开始计数）
- **列**: DM码所在的列号（从1开始计数）
- **G间距**: 条码间距（每隔多少个方格放置一个DM码）
- **S棋盘格大小**: 单个棋盘格的物理尺寸（单位：mm）

**Data Matrix码特征**：DM码边缘有两条实体线（L型定位图案），这是Data Matrix码的标准特征。

## 输出文件

所有生成的文件会自动保存到 `output/` 文件夹中（如果文件夹不存在会自动创建）。

- **图像文件**: 生成的棋盘格图像（PNG格式）
  - 默认文件名：`output/checkerboard_with_dm.png`
- **配置文件**: 自动生成的JSON配置文件
  - 默认文件名：`output/checkerboard_with_dm_config.json`
  - 包含所有参数和坐标系信息

## 依赖库

- `numpy>=1.21.0`: 数值计算
- `opencv-python>=4.5.0`: 图像处理
- `Pillow>=8.0.0`: 图像处理
- `pylibdmtx>=0.1.9`: Data Matrix码生成（真正的DM码，边缘有两条实体线）

## 项目结构

```
CalcBoard/
├── checkerboard_generator.py  # 主程序文件
├── test_generator.py          # 测试脚本
├── README.md                  # 说明文档
├── requirements.txt           # 依赖列表
└── output/                    # 输出文件夹（自动创建）
    ├── checkerboard_with_dm.png
    └── checkerboard_with_dm_config.json
```

## 使用示例

### 示例1：生成标准棋盘格

```python
from checkerboard_generator import generate_checkerboard_simple

# 生成20mm方格，25x25的棋盘格
generate_checkerboard_simple(
    square_size=20.0,
    rows=25,
    cols=25,
    dm_spacing=5,
    dm_start_position=4
)
```

### 示例2：生成小尺寸棋盘格

```python
# 生成10mm方格，15x15的棋盘格
generate_checkerboard_simple(
    square_size=10.0,
    rows=15,
    cols=15,
    dm_spacing=5,
    dm_start_position=4
)
```

### 示例3：查看生成的DM码信息

运行程序后，会在控制台输出所有生成的DM码信息：

```
棋盘格已生成并保存到: checkerboard_with_dm.png
配置文件已保存到: checkerboard_with_dm_config.json

共生成 4 个DM码:
  ID:1, 坐标:(40.0, 40.0)mm, 位置:第5行第5列 (占据2x2方格: 行5-6, 列5-6), 条码内容: 5,5,G5S20.0
  ID:2, 坐标:(90.0, 40.0)mm, 位置:第5行第10列 (占据2x2方格: 行5-6, 列10-11), 条码内容: 5,10,G5S20.0
  ...
```

## 注意事项

1. **DM码生成**: 本工具使用 `pylibdmtx` 生成真正的Data Matrix码（DM码），DM码边缘有两条实体线（L型定位图案）
2. **坐标信息格式**: `行,列,G间距S棋盘格大小`（例如：`5,5,G5S20.0`）
3. **DM码尺寸**: 每个DM码占据2x2个方格，会自动清除背景
4. **边框大小**: DM码边框自适应，大小为1个内部模块大小
5. **边界检查**: 超出棋盘格边界的DM码会被自动跳过
6. **物理坐标系**: 默认原点在左上角，X轴向右，Y轴向下
7. **棋盘格颜色**: 第一个方格（左上角）为黑色，形成标准的黑白交替图案
8. **输出权限**: 确保输出目录有写入权限
9. **依赖安装**: 如果 `pylibdmtx` 安装失败，程序会显示警告并使用占位符

## 测试

运行测试脚本验证功能：

```bash
python test_generator.py
```

## 许可证

MIT License
