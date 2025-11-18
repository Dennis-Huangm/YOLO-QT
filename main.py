import os
import random
import yaml
from mainwindow import Ui_MainWindow
import cv2
import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import *
from ultralytics.utils.files import increment_path
from pathlib import Path
import time
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cvimg_to_qtimg(cvimg):
    """
    将OpenCV图像转换为PyQt图像
    """
    height, width, depth = cvimg.shape  # 返回（高、宽、通道数）depth为3
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)  # cvimg.data为图片存放的地址，cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
    cvimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
    # 将cv类型的图片转换为Qpixmap显示在Qlabel上出现扭曲的问题
    '''QImage(img.data, img.shape[1], img.shape[0],QImage.Format_RGB888)
       应当改为img = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3,QImage.Format.Format_BGR888)
       可避免应参数不完全导致的错误'''
    return cvimg


class UiMain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(UiMain, self).__init__(parent)
        self.model = None  # 模型
        self.setupUi(self)

        self.OpenButton.setVisible(False)  # 导入视频、图片
        self.OpenButton.clicked.connect(self.loadimage)

        self.EndButton.clicked.connect(self.close)  # 退出程序

        self.fname = None  # 文件名
        self.weight_path = 'yolov8n.pt'

        self.ReButton.setVisible(False)  # 重新预测
        self.ReButton.clicked.connect(self.rework)

        self.starButton.clicked.connect(self.model_star)  # 初始化模型

        self.stopButton.setVisible(False)  # 结束检测
        self.stopButton.clicked.connect(self.stop)

        self.PauseButton.setVisible(False)  # 暂停检测
        self.PauseButton.clicked.connect(self.pause)

        self.ContinueButton.setVisible(False)  # 继续检测
        self.ContinueButton.clicked.connect(self.work)

        self.cameraButton.setVisible(False)  # 启动摄像头
        self.cameraButton.clicked.connect(self.camera)
        self.weightButton.clicked.connect(self.weight)

        self.labelButton.clicked.connect(self.start_label)
        self.labelbutton2.clicked.connect(self.label_f)
        self.safeButton.clicked.connect(self.safe)

        self.pushButton.clicked.connect(self.pre_split)
        self.splitButton.clicked.connect(self.split_data)
        self.yamlButton.clicked.connect(self.create_yaml)

        self.epoch_Slider.valueChanged.connect(self.epoch_f)
        self.batch_Slider.valueChanged.connect(self.batch_f)
        self.workers_Slider.valueChanged.connect(self.workers_f)
        self.train_Button.clicked.connect(self.train)

        self.batch = 16
        self.epoch = 100
        self.workers = 0

        self.vid_writer = None
        self.predefined_classes = None
        self.yaml_path = None

        self.stop = False
        self.category = None
        self.label_directory = None
        self.data_to_split = None
        self.f_present = 0  # 与暂停开始播放有关
        self.f_pause = 0
        self.t1 = 0

        self.ROOT = Path(__file__).resolve().parents[0]

    def weight(self):
        self.weight_path, _ = QFileDialog.getOpenFileName(self, '请选择权重文件', '.', '权重文件(*.pt)')

    def model_star(self):
        if not self.weight_path:
            self.weight_path = 'yolov8n.pt'
        self.model = YOLO(str(self.weight_path))
        self.weightButton.setVisible(False)
        self.OpenButton.setVisible(True)
        self.cameraButton.setVisible(True)
        self.textEdit.setText(f"模型初始化成功:{self.weight_path}")
        self.starButton.setVisible(False)
        self.category = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def camera(self):
        self.fname = 0
        self.statusbar.showMessage('已启动摄像头', 10000)
        self.work()

    def loadimage(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, '请选择图片', '.', '图像文件(*.jpg *.jpeg *.png, *.mp4 *.flv)')
        if self.fname:
            self.statusbar.showMessage(f'已成功打开文件{self.fname}', 10000)
            self.work()
        else:
            self.textEdit.setText("打开文件失败")

    def rework(self):
        self.f_pause = 0
        result = QMessageBox.question(self, "注意：", "您真的要重新检测吗？", QMessageBox.Yes | QMessageBox.No,
                                      QMessageBox.No)
        if result == QMessageBox.Yes:
            self.work()
            self.statusbar.showMessage(f'已重新预测文件{self.fname}', 10000)

    def work(self):
        if self.fname or self.fname == 0:
            # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)] #产生80种颜色
            self.t1 = time.time()

            if self.fname == 0 or self.fname.endswith(".mp4") or self.fname.endswith("*.flv"):  # 可能有点问题
                self.cameraButton.setVisible(False)
                self.ReButton.setVisible(False)
                self.EndButton.setVisible(False)
                self.OpenButton.setVisible(False)
                self.ContinueButton.setVisible(False)
                self.PauseButton.setVisible(True)
                self.stopButton.setVisible(True)
                cap = cv2.VideoCapture(self.fname)  # 参数是0时，默认使用电脑自带的摄像头
                # cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) #设置缓存区的大小 !!!
                self.f_present = 0
                self.stop = False

                if self.checkBox.isChecked():
                    self.vid_writer = self.video_writer(cap)

                while True:  # while cap.isOpened():
                    if self.stop:
                        break

                    cap.grab()  # 跳一帧
                    bool_value, img0 = cap.retrieve()  # 读取每张图片
                    '''while cap.isOpened():  异步与同步的区别
                         flag, frame = cap.read()'''
                    # grab:方法/函数从视频文件或摄像机抓取下一帧，并在成功的情况下返回true（非零）抓取的意思，即是将read换成grab加快取流进程的读取速度
                    # retrieve:读取：解码并返回下一个视频帧
                    self.f_present += 1
                    if self.f_present <= self.f_pause:
                        continue

                    if not bool_value:
                        self.f_pause = 0
                        self.OpenButton.setVisible(True)
                        self.ReButton.setVisible(True)
                        self.stopButton.setVisible(False)
                        self.PauseButton.setVisible(False)
                        self.EndButton.setVisible(True)
                        self.cameraButton.setVisible(True)
                        break

                    self.detect(img0)

                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        cap.release()
                        cv2.destroyAllWindows()
                        raise StopIteration

            else:
                pic = cv2.imread(self.fname)
                self.detect(pic)
                self.ReButton.setVisible(True)
                # QMessageBox.warning(self, "警告", "消防通道被占用！！", QMessageBox.Yes)

        if self.checkBox.isChecked():
            self.statusbar.showMessage('已保存，若需查看请结束程序后再进行查看', 10000)

    def detect(self, img0):
        result = self.model(img0)  # 预测单张图片
        boxes = result[0].boxes
        plot_args = dict(line_width=None, boxes=True, conf=True, labels=True)
        plotted_img = result[0].plot(**plot_args)

        result_qt = cvimg_to_qtimg(plotted_img)
        jpg = QtGui.QPixmap(result_qt).scaled(self.Imglabel.width(), self.Imglabel.height())
        # 使用scaled方法把 pix_img 的分辨率(图片的大小)设置为固定像素

        t2 = time.time()
        t = t2 - self.t1
        self.t1 = t2

        pred = ''
        self.Imglabel.setPixmap(jpg)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(len(result[0].boxes.data)):
            boxes_cls = self.category[int(boxes.cls[i].cpu().numpy())]
            pred += '坐标' + str(boxes.xyxy[i].cpu().numpy()) + ':' + str(boxes_cls) + '\n\n'
        pred += "所用时间:" + str(t)
        self.textEdit.setText(pred)

        if self.vid_writer:
            self.vid_writer.write(plotted_img)

        if self.checkBox.isChecked() and (str(self.fname).endswith('.jpg') or (str(self.fname).endswith('.png'))):
            filename = os.path.basename(self.fname)
            save_dir = increment_path(Path(self.ROOT / 'runs' / 'detect') / 'predict', exist_ok=False, mkdir=True)
            save_path = str(Path(str(save_dir) + '/' + str(filename)))

            cv2.imwrite(save_path, plotted_img)

    def stop(self):
        self.stop = True
        self.f_pause = 0
        self.stopButton.setVisible(False)
        self.OpenButton.setVisible(True)
        if self.fname != 0:
            self.ReButton.setVisible(True)
        self.stopButton.setVisible(False)
        self.PauseButton.setVisible(False)
        self.EndButton.setVisible(True)
        self.ContinueButton.setVisible(False)
        self.cameraButton.setVisible(True)

    def pause(self):
        self.stop = True
        if self.fname != 0:
            self.f_pause = self.f_present
        self.ContinueButton.setVisible(True)
        self.PauseButton.setVisible(False)

    def close(self):
        result = QMessageBox.question(self, "注意：", "您真的要结束程序吗？", QMessageBox.Yes | QMessageBox.No,
                                      QMessageBox.No)
        if result == QMessageBox.Yes:
            app_ = QApplication.instance()
            app_.quit()

    def video_writer(self, cap):
        filename = os.path.basename(self.fname) if self.fname else 'camera'
        save_dir = increment_path(Path(self.ROOT / 'runs' / 'detect') / 'predict', exist_ok=False, mkdir=True)
        save_path = str(Path(str(save_dir) + '/' + str(filename)).with_suffix('.mp4'))
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)
        return vid_writer

    def label_f(self):
        self.label_directory = QFileDialog.getExistingDirectory(self, "选取文件夹", "C:/")
        if self.label_directory:
            self.lineEdit.setText(self.label_directory)

    def safe(self):
        if self.label_directory:
            num = 1
            classes = self.classEdit.toPlainText().strip()
            classes_num = self.comboBox.currentText()
            for string in classes:
                if string == '\n':
                    num += 1
            if num != int(classes_num) or num == 1 and not classes:
                QMessageBox.question(self, "注意：", "类别数量不匹配", QMessageBox.Ok)
            else:
                os.chdir(self.label_directory)
                with open('../predefined_classes.txt', 'w') as file:
                    file.write(classes)
                self.predefined_classes = os.path.abspath('../predefined_classes.txt')
                self.statusbar.showMessage('已自动生成预标注文件', 3000)
        else:
            QMessageBox.question(self, "注意：", "未选择图片所在目录", QMessageBox.Ok)

    def start_label(self):
        if self.label_directory:
            os.chdir(self.label_directory)
            if os.path.exists('../predefined_classes.txt'):
                self.predefined_classes = os.path.abspath('../predefined_classes.txt')
            try:
                os.mkdir('../labels')
            except FileExistsError:
                pass
            os.chdir((os.path.dirname(
                __file__)))  # ‘__file__'这个参数代表的是python解释器正在执行的脚本文件，如果直接在命令行运行上面两行代码，则python解释器找不到正在执行的脚本文件.
            # 在pycharm等IDE中运行程序的话，默认设置启动路径为项目根目录，所以直接执行是没有问题的。所以如果在cmd中启动项目的话，需要将本项目的路径加入系统变量中
            os.system(f'cd labelimg&&python labelimg.py {self.label_directory} {self.predefined_classes} ')
        else:
            QMessageBox.question(self, "注意：", "未选择图片所在目录", QMessageBox.Ok)

    def pre_split(self):
        self.data_to_split = QFileDialog.getExistingDirectory(self, "选取文件夹", "C:/")
        if self.data_to_split:
            self.lineEdit_2.setText(self.data_to_split)

    def split_data(self):
        if self.data_to_split:
            os.chdir(self.data_to_split)
            data_list = os.listdir(self.data_to_split)
            num = len(data_list)
            rate = 0.1
            val = random.sample(data_list, int(num * rate))
            train = list(set(data_list) - set(val))
            with open('../train.txt', 'w') as train_file:
                for i in train:
                    train_file.write(os.path.abspath(i) + '\n')
            with open('../val.txt', 'w') as val_file:
                for i in val:
                    val_file.write(os.path.abspath(i) + '\n')
            self.statusbar.showMessage('已生成训练集、测试集的索引文件', 5000)
        else:
            QMessageBox.question(self, "注意：", "未选择图片所在目录", QMessageBox.Ok)

        if not str(self.data_to_split).endswith('images') and self.data_to_split:
            QMessageBox.question(self, "注意：", "图片目录名必须为images才可正常训练，请先退出程序后更改", QMessageBox.Ok)

    def create_yaml(self):
        if self.data_to_split:
            os.chdir(self.data_to_split)
            if os.path.exists('../predefined_classes.txt') and os.path.exists('../train.txt') and \
                    os.path.exists('../val.txt'):
                num = 0
                class_dic = {}
                with open('../predefined_classes.txt', 'r') as file:
                    classes = file.readlines()
                for i in range(len(classes)):
                    if classes[i].strip():
                        num += 1
                        class_dic[i] = classes[i].strip()
                desired_caps = {
                    'train': os.path.abspath('../train.txt'),
                    'val': os.path.abspath('../val.txt'),
                    'names': class_dic,
                    'nc': num
                }
                os.chdir((os.path.dirname(__file__)))
                self.yaml_path = 'tmp.yaml'
                with open(self.yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(desired_caps, f)
                self.statusbar.showMessage(f'已自动生成yaml文件{os.path.abspath(self.yaml_path)}', 5000)
            else:
                QMessageBox.question(self, "注意：", "该数据集中缺少训练所需文件，请先仔细进行标注", QMessageBox.Ok)
        else:
            QMessageBox.question(self, "注意：", "请先选择需要训练的数据集", QMessageBox.Ok)

    def epoch_f(self):
        self.epoch = self.epoch_Slider.value()
        self.epoch_label.setText(f'训练的回合数：{str(self.epoch)}')

    def batch_f(self):
        self.batch = self.batch_Slider.value()
        self.batch_label.setText(f'训练的批次大小：{str(self.batch)}')

    def workers_f(self):
        self.workers = self.workers_Slider.value()
        self.workers_label.setText(f'载入数据的进程数：{str(self.workers)}')

    def train(self):
        os.chdir((os.path.dirname(__file__)))
        save = self.checkBox_2.isChecked()
        model = YOLO(str(self.model_comboBox.currentText()))
        if self.yaml_comboBox.currentIndex() == 0 and not self.yaml_path:
            QMessageBox.question(self, "注意：", "请先点击上方摁钮，创造一个自定义的yaml文件", QMessageBox.Ok)
            return 0
        elif self.yaml_comboBox.currentIndex() == 0 and self.yaml_path:
            path = self.yaml_path
        else:
            path = str(self.yaml_comboBox.currentText())
        model.train(data=path, epochs=self.epoch, batch=self.batch, workers=self.workers, save=save,
                    optimizer=self.optimizer_comboBox.currentText(), project="runs", name="train/exp")
        model.val(project="runs", name="val/exp")
        QMessageBox.question(self, "注意：", "模型训练完成！", QMessageBox.Ok)


if __name__ == '__main__':
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    ui = UiMain()
    ui.show()
    sys.exit(app.exec_())
