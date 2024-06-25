from PIL import Image

def remove_white_outline(image_path, tolerance=230, save_path="output.png"):
    # 打开图片
    img = Image.open(image_path)
    img = img.convert("RGBA")  # 确保图片有透明通道

    # 加载图片数据
    datas = img.getdata()

    # 新的图片数据
    new_data = []
    for item in datas:
        # 将接近白色的像素改为透明
        if item[0] > tolerance and item[1] > tolerance and item[2] > tolerance:
            new_data.append((255, 255, 255, 0))  # 完全透明
        else:
            new_data.append(item)

    # 更新图片数据并保存
    img.putdata(new_data)
    img.save(save_path)

# 调用函数
remove_white_outline("resized_image-removebg-preview.png")
