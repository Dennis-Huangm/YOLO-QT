# Denis
# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QStatusBar, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置主窗口
        self.setWindowTitle('QStatusBar 字体颜色示例')
        self.setGeometry(100, 100, 400, 200)

        # 创建状态栏
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)

        # 创建 QLabel 并设置样式表以改变字体颜色
        self.label = QLabel('这是一个红色字体的消息')
        self.label.setStyleSheet('color: red;')  # 将字体颜色设置为红色
        self.statusBar.addWidget(self.label, 1)  # 1 表示拉伸因子

        # 添加另一个 QLabel 以显示不同颜色
        self.label2 = QLabel('这是一个蓝色字体的消息')
        self.label2.setStyleSheet('color: blue;')  # 将字体颜色设置为蓝色
        self.statusBar.addWidget(self.label2, 1)  # 1 表示拉伸因子
        self.statusBar.removeWidget(self.label1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())