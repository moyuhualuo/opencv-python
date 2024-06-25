import json

# 初始的 Hand_dict
Hand_dict = {
    '01000': 'One',
    '01100': 'Two',
    '11100': 'Three',
    '01111': 'Four',
    '11111': 'Five',
    '01110': 'Six',
    '01101': 'Seven',
    '01011': 'Eight',
    '00111': 'Nine',
    '10000': 'Ten',
    '00000': 'Zero',
    '01001': 'peace'
}

# 保存 Hand_dict 到文件
def save_hand_dict(filename):
    with open(filename, 'w') as f:
        json.dump(Hand_dict, f)

# 添加新手势到 Hand_dict
def add_to_hand_dict(no, name):
    if no in Hand_dict:
        print(f"Entry '{no}' already exists in Hand_dict. Change failed'.")
    else:
        Hand_dict[no] = name
        save_hand_dict('hand_dict.json')
        print(f"Added '{name}' with code '{no}' to Hand_dict and saved to file.")

