import rasterio
import os

# 1. 检查文件是否存在
image_path = '/root/autodl-tmp/ortho_tif.tif'
if not os.path.exists(image_path):
    print(f"错误：文件不存在 - {image_path}")
else:
    print(f"文件存在，大小：{os.path.getsize(image_path) / 1024 / 1024:.2f} MB")
    
    # 2. 尝试读取文件
    try:
        with rasterio.open(image_path) as src:
            # 打印影像基本信息
            print(f"影像宽度: {src.width}")
            print(f"影像高度: {src.height}")
            print(f"波段数: {src.count}")
            print(f"投影信息: {src.crs}")
            print("文件读取成功！")
    except Exception as e:
        print(f"读取失败，错误信息：{e}")
        # 尝试用 GDAL 检查详细错误
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            ds = gdal.Open(image_path)
            if ds is None:
                print("GDAL 也无法打开该文件，确认文件损坏或格式不支持")
            else:
                print("GDAL 可以打开，可能是 rasterio 配置问题")
        except ImportError:
            print("未安装 GDAL，无法进一步检查")