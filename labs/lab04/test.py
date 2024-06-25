import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton
import save


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('获取两个输入值示例')
        self.setGeometry(100, 100, 400, 200)  # 设置窗口位置和大小

        layout = QVBoxLayout(self)

        # 输入框1
        self.input1 = QLineEdit(self)
        self.input1.setPlaceholderText('请输入要改的编号')
        layout.addWidget(self.input1)

        # 输入框2
        self.input2 = QLineEdit(self)
        self.input2.setPlaceholderText('请输入要设定的含义')
        layout.addWidget(self.input2)

        # 按钮
        button = QPushButton('获取输入值', self)
        button.clicked.connect(self.onButtonClicked)  # 连接按钮点击事件
        layout.addWidget(button)

        self.setLayout(layout)

    def onButtonClicked(self):
        value1 = self.input1.text()
        value2 = self.input2.text()
        print(f'Button clicked with values: {value1}, {value2}')
        save.add_to_hand_dict(value1, value2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
