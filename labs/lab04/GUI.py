import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QLineEdit
from PyQt5.QtGui import QFont
import subprocess
from PyQt5.QtGui import QIcon, QPixmap
import json

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyQt Style Sheet Example')
        self.setGeometry(100, 100, 1200, 700)

        # 使用样式表设置背景颜色和字体
        self.setStyleSheet('''
            background-color: #0D0D0D;  /* 设置背景颜色为浅灰色 */
            font-family: "playwrite NZ", cursive;
            font-size: 36px;  /* 设置字体大小为16像素 */
            color: #EDF2E9;  /* 设置文字颜色为深灰色 */
        ''')
        vbox = QVBoxLayout()
        # 创建一个标签并设置样式
        label01 = QLabel('Hand Detector', self)
        label01.setStyleSheet('''
            color: #EDF2E9;  /* 设置文字颜色为白色 */
            background-color: #0D0D0D;  /* 设置背景颜色为深蓝色 */
            padding: 10px;  /* 设置内边距为10像素 */
            border-radius: 25px;  /* 设置圆角边框 */
            font-family: "playwrite NZ", cursive;
            font-size: 36px;
            font-weight: 400;
            font-style: normal;
            font-optical-sizing: auto;
        ''')
        label01.move(10, 10)

        label02 = QLabel('Opencv Lab', self)
        label02.setStyleSheet('''
                    color: #0D0D0D;  /* 设置文字颜色为白色 */
                    background-color: #EDF2E9;  /* 设置背景颜色为深蓝色 */
                    padding: 10px;  /* 设置内边距为10像素 */
                    border-radius: 20px;  /* 设置圆角边框 */
                    font-family: "playwrite NZ", cursive;
                    font-size: 24px;
                    font-weight: 400;
                    font-style: normal;
                    font-optical-sizing: auto;
                ''')
        label02.move(110, 70)

        label03 = QLabel('Created by Liu', self)
        label03.setStyleSheet('''
                            color: #262626;  /* 设置文字颜色为白色 */
                            background-color: #F2E205;  /* 设置背景颜色为深蓝色 */
                            padding: 10px;  /* 设置内边距为10像素 */
                            border-radius: 20px;  /* 设置圆角边框 */
                            font-family: "playwrite NZ", cursive;
                            font-size: 24px;
                            font-weight: 400;
                            font-style: normal;
                            font-optical-sizing: auto;
                        ''')
        label03.move(60, 120)

        label04 = QLabel('HD.目前存在4个功能:\n Function1:\n     单张图片识别 \n Function2:\n     视频手势跟踪 \n Function3:\n     数字ALS手势识别 \n Function4:\n     添加手势编号', self)
        label04.setStyleSheet('''
                                    color: #EDF2E9;  /* 设置文字颜色为白色 */
                                    background-color: #025159;  /* 设置背景颜色为深蓝色 */
                                    padding: 10px;  /* 设置内边距为10像素 */
                                    border-radius: 10px;  /* 设置圆角边框 */
                                    font-family: "playwrite NZ", cursive;
                                    font-size: 24px;
                                    font-weight: 400;
                                    font-style: normal;
                                    font-optical-sizing: auto;
                                ''')
        label04.move(10, 170)

        button1 = QPushButton('Function1', self)
        button1.setStyleSheet('''
                    background-color: #024959;  /* 设置按钮背景颜色为蓝色 */
                    color: white;  /* 设置文字颜色为白色 */
                    border-style: outset;  /* 设置边框样式为凸起 */
                    border-width: 2px;  /* 设置边框宽度为2像素 */
                    border-radius: 10px;  /* 设置边框圆角半径为10像素 */
                    padding: 60px;  /* 设置内边距为5像素 */
                ''')
        button1.clicked.connect(self.onButton1Clicked)  # 连接按钮点击事件
        button1.move(400, 100)

        button2 = QPushButton('Function2', self)
        button2.setStyleSheet('''
                            background-color: #024959;  /* 设置按钮背景颜色为蓝色 */
                            color: white;  /* 设置文字颜色为白色 */
                            border-style: outset;  /* 设置边框样式为凸起 */
                            border-width: 2px;  /* 设置边框宽度为2像素 */
                            border-radius: 10px;  /* 设置边框圆角半径为10像素 */
                            padding: 60px;  /* 设置内边距为5像素 */
                        ''')
        button2.clicked.connect(self.onButton2Clicked)  # 连接按钮点击事件
        button2.move(800, 100)

        button3 = QPushButton('Function3', self)
        button3.setStyleSheet('''
                                    background-color: #024959;  /* 设置按钮背景颜色为蓝色 */
                                    color: white;  /* 设置文字颜色为白色 */
                                    border-style: outset;  /* 设置边框样式为凸起 */
                                    border-width: 2px;  /* 设置边框宽度为2像素 */
                                    border-radius: 10px;  /* 设置边框圆角半径为10像素 */
                                    padding: 60px;  /* 设置内边距为5像素 */
                                ''')
        button3.clicked.connect(self.onButton3Clicked)  # 连接按钮点击事件
        button3.move(400, 350)

        button4 = QPushButton('Function4', self)
        button4.setStyleSheet('''
                                            background-color: #024959;  /* 设置按钮背景颜色为蓝色 */
                                            color: white;  /* 设置文字颜色为白色 */
                                            border-style: outset;  /* 设置边框样式为凸起 */
                                            border-width: 2px;  /* 设置边框宽度为2像素 */
                                            border-radius: 10px;  /* 设置边框圆角半径为10像素 */
                                            padding: 60px;  /* 设置内边距为5像素 */
                                        ''')
        button4.clicked.connect(self.onButton4Clicked)  # 连接按钮点击事件
        button4.move(800, 350)

    def onButton1Clicked(self):
        options = QFileDialog.Options()
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                    "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if image_path:
            print(f"Selected image path: {image_path}")
            try:
                subprocess.run(["python", "HandImage.py", image_path])
            except Exception as e:
                print(f"Error calling external script: {e}")

    def onButton2Clicked(self):
        print('Button 2 clicked')
        try:
            subprocess.run(["python", "Hand1.0.py"])
        except Exception as e:
            print(f"Error calling external script: {e}")

    def onButton3Clicked(self):
        print('Button 3 clicked')
        try:
            subprocess.run(["python", "HandVideo.py"])
        except Exception as e:
            print(f"Error calling external script: {e}")

    def onButton4Clicked(self):
        print('Button 4 clicked')
        try:
            subprocess.run(["python", "test.py"])
        except Exception as e:
            print(f"Error calling external script: {e}")



if __name__ == '__main__':
    try:
        with open('hand_dict.json', 'r') as f:
            Hand_dict = json.load(f)
    except FileNotFoundError:
        Hand_dict = {}

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
