import json
import os

import javalang
from tqdm import tqdm


def getStartEnd(find_node, tree):
    start = None
    end = None
    for path, node in tree:
        if start is not None and find_node not in path:
            end = node.position
            return start, end
        if start is None and node == find_node:
            start = node.position
    return start, end


def getMethod(start, end, data):
    if start is None:
        return ""
    eos = None
    if end is not None:
        eos = end.line - 1

    lines = data.splitlines(True)
    method = "".join(lines[start.line:eos])
    method = lines[start.line - 1] + method

    if end is None:
        left = method.count("{")
        right = method.count("}")
        if right - left == 1:
            p = method.rfind("}")
            method = method[:p]

    return method


def getAllMethods(file_path):
    methods = []
    id = 1
    with open(file_path, 'r') as f:
        data = f.read()
    try:
         tree = javalang.parse.parse(data)
         for _,node in tree.filter(javalang.tree.MethodDeclaration):
             start,end=getStartEnd(node,tree)

             file_methods = {
                 "method_id": id,
                 "method_data": getMethod(start, end, data)
             }
             id += 1

             methods.append(file_methods)
    except Exception as e:
        print(e)

    return methods


def getMethodsDirectory(directory):
    all_methods = []
    id = 1
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.startswith("test") and file.endswith('.java'):
                file_path = os.path.join(root, file)
                file_data = getAllMethods(file_path)

                if len(file_data) == 0:
                    continue

                file_methods = {
                    "file_id": id,
                    "file_path": file_path,
                    "file_data": file_data
                }
                id += 1

                all_methods.append(file_methods)

    return all_methods


java_folders = "bc-java"

all_methods = []
id = 1
for subdir in tqdm(os.listdir(java_folders)):
    project_dir = os.path.join(java_folders, subdir)
    if os.path.isdir(project_dir):
        folder_data = getMethodsDirectory(project_dir)

        if len(folder_data) == 0:
            continue

        subfolder = {
            "folder_id": id,
            "folder_path": project_dir,
            "folder_data": folder_data
        }
        id += 1

        all_methods.append(subfolder)

final_data = "final_method_data.json"

with open(final_data, 'w') as json_file:
    json.dump(all_methods, json_file, indent=4)

print("done")
