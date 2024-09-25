import os
import javalang

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
    with open(file_path, 'r') as f:
        data = f.read()
    try:
         tree = javalang.parse.parse(data)
         for _,node in tree.filter(javalang.tree.MethodDeclaration):
             start,end=getStartEnd(node,tree)
             methods.append(getMethod(start, end, data))
    except Exception as e:
        print(e)

    return methods

def getMethodsDirectory(directory):
    all_methods = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                methods = getAllMethods(file_path)
                all_methods.extend(methods)
    return all_methods


java_folders = "/Users/babu/Documents/csci_680_Ai_swe/assignment_01_data/all_repo"

all_methods = []

for subdir in os.listdir(java_folders):
    project_dir = os.path.join(java_folders, subdir)
    if os.path.isdir(project_dir):
        methods = getMethodsDirectory(project_dir)
        all_methods.extend(methods)

final_data = "/Users/babu/Documents/csci_680_Ai_swe/assignment_01_data/final_method_data.txt"

with open(final_data, 'w', encoding='utf-8') as f:
    for method in all_methods:
        f.write(f"{method}\n\n")

print("done")