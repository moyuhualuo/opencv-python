import os
import shutil


def classify_files_by_prefix(directory):
    directory = os.path.normpath(directory)
    if not os.path.isdir(directory):
        raise IOError(f"{directory} is not a valid directory.")

    objects = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                prefix = file.split('_')[0].lower()  # 提取文件名的前缀，并转换为小写
                if prefix not in objects:
                    objects[prefix] = []
                objects[prefix].append(os.path.join(root, file))

    return objects


def move_files_to_directory(objects, target_directory):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for prefix, files in objects.items():
        prefix_dir = os.path.join(target_directory, prefix)
        os.makedirs(prefix_dir, exist_ok=True)
        for file in files:
            shutil.move(file, os.path.join(prefix_dir, os.path.basename(file)))


# 示例用法
train_directory = r'D:\opencv\hws\hw04\archive\train_zip\train'
classified_objects = classify_files_by_prefix(train_directory)

# 将文件移动到分类的目标文件夹
target_directory = r'D:\opencv\hws\hw04\archive\classified_train'
move_files_to_directory(classified_objects, target_directory)

print("文件分类完成。")
