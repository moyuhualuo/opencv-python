import json

def load_hand_dict(filename):
    with open(filename, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    hand_dict = load_hand_dict('hand_dict.json')
    print(hand_dict)  # 可以打印检查加载的内容
