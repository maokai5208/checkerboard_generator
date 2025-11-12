"""
快速测试脚本 - 验证棋盘格生成工具是否正常工作
"""

import sys
import os

# 设置Windows控制台编码为UTF-8
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

def test_imports():
    """测试必要的库是否已安装"""
    print("检查依赖库...")
    missing = []
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError:
        missing.append("numpy")
        print("✗ numpy - 未安装")
    
    try:
        import cv2
        print("✓ opencv-python")
    except ImportError:
        missing.append("opencv-python")
        print("✗ opencv-python - 未安装")
    
    try:
        from PIL import Image
        print("✓ Pillow")
    except ImportError:
        missing.append("Pillow")
        print("✗ Pillow - 未安装")
    
    try:
        from pylibdmtx.pylibdmtx import encode
        print("✓ pylibdmtx")
    except ImportError:
        missing.append("pylibdmtx")
        print("✗ pylibdmtx - 未安装（需要用于生成Data Matrix码）")
    
    if missing:
        print(f"\n缺少以下库: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True


def test_generator():
    """测试棋盘格生成器"""
    print("\n测试棋盘格生成器...")
    
    try:
        from checkerboard_generator import CheckerboardGenerator
        
        # 创建生成器
        generator = CheckerboardGenerator(
            rows=5,
            cols=4,
            square_size=20.0,
            image_size=(1000, 1200),
            margin=50
        )
        
        print("✓ CheckerboardGenerator 创建成功")
        
        # 测试生成棋盘格
        img = generator.generate_checkerboard()
        print(f"✓ 棋盘格生成成功，尺寸: {img.shape}")
        
        # 测试生成DM码
        test_data = "ID:1,X:0,Y:0"
        dm_img = generator.generate_dm_code(test_data, size=100)
        print(f"✓ DM码生成成功，尺寸: {dm_img.shape}")
        
        # 测试保存
        test_output = "test_checkerboard.png"
        if os.path.exists(test_output):
            os.remove(test_output)
        
        # 使用简化的接口进行测试
        from checkerboard_generator import generate_checkerboard_simple
        
        # 生成测试棋盘格
        generate_checkerboard_simple(
            square_size=20.0,
            rows=5,
            cols=4,
            dm_spacing=2,
            dm_start_position=1,
            output_path=test_output
        )
        
        if os.path.exists(test_output):
            print(f"✓ 文件保存成功: {test_output}")
            config_file = test_output.replace('.png', '_config.json')
            if os.path.exists(config_file):
                print(f"✓ 配置文件保存成功: {config_file}")
            # 注意：测试文件已生成，不会被删除，您可以查看生成的文件
        else:
            print(f"✗ 文件保存失败: {test_output}")
            return False
        
        print("\n✓ 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("棋盘格生成工具 - 测试脚本")
    print("=" * 50)
    
    # 测试导入
    if not test_imports():
        sys.exit(1)
    
    # 测试生成器
    if not test_generator():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

