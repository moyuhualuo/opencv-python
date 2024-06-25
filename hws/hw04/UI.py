import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import test01

class FruitRecognitionUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fruit Recognition")
        self.setGeometry(100, 100, 600, 600)

        self.layout = QVBoxLayout()

        # 标题标签
        self.title_label = QLabel("Fruit Recognition System", self)
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        # 图片显示标签
        self.image_label = QLabel("No image loaded", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid black;")
        self.layout.addWidget(self.image_label)

        # 按钮布局
        self.button_layout = QHBoxLayout()

        # 加载图片按钮
        self.load_button = QPushButton("Load Image", self)
        self.load_button.setStyleSheet("background-color: lightblue; font-size: 16px;")
        self.load_button.clicked.connect(self.load_image)
        self.button_layout.addWidget(self.load_button)

        # 识别水果按钮
        self.recognize_button = QPushButton("Recognize Fruit", self)
        self.recognize_button.setStyleSheet("background-color: lightgreen; font-size: 16px;")
        self.recognize_button.clicked.connect(self.recognize_fruit)
        self.button_layout.addWidget(self.recognize_button)

        self.layout.addLayout(self.button_layout)

        # 识别结果标签
        self.result_label = QLabel("", self)
        self.result_label.setFont(QFont("Arial", 16))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

        self.image_path = None

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.jpg *.jpeg *.png)",
                                                   options=options)

        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
            self.result_label.setText("")

    def recognize_fruit(self):
        if self.image_path:
            # 这里可以加入你识别水果的代码，比如加载模型、提取特征等
            # 假设识别结果是 "Apple"

            predicted_category = test01.function(self.image_path)
            #print(f"预测类别: {}"
            result = predicted_category
            self.result_label.setText(f"Recognition Result: {result}")
        else:
            self.result_label.setText("No image loaded")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FruitRecognitionUI()
    window.show()
    sys.exit(app.exec_())
