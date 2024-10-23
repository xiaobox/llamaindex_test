import pkg_resources
import re
from pathlib import Path

def get_installed_version(package_name):
    """获取已安装包的版本号"""
    try:
        # 处理类似 httpx[socks] 这样的包名
        base_package = package_name.split('[')[0]
        return pkg_resources.get_distribution(base_package).version
    except pkg_resources.DistributionNotFound:
        return None

def update_requirements(file_path):
    """更新 requirements.txt 文件中的版本号"""
    requirements_path = Path(file_path)
    if not requirements_path.exists():
        print(f"找不到文件：{file_path}")
        return

    # 读取原文件内容
    with open(requirements_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        # 跳过注释和空行
        if line.startswith('#') or not line.strip():
            updated_lines.append(line)
            continue

        # 提取包名
        package_match = re.match(r'^([^=<>]+)', line.strip())
        if package_match:
            package_name = package_match.group(1).strip()
            version = get_installed_version(package_name)
            
            if version:
                # 保持原有的注释（如果有的话）
                comment = line.split('#', 1)[1].strip() if '#' in line else ''
                new_line = f"{package_name}=={version}"
                if comment:
                    new_line += f"  # {comment}"
                new_line += '\n'
                updated_lines.append(new_line)
                print(f"更新 {package_name} 版本为 {version}")
            else:
                updated_lines.append(line)
                print(f"警告：未找到包 {package_name} 的安装信息")
        else:
            updated_lines.append(line)

    # 写入更新后的内容
    with open(requirements_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)

    print("\n更新完成！")

if __name__ == "__main__":
    requirements_file = "requirements.txt"
    update_requirements(requirements_file)
