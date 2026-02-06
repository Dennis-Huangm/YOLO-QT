import os
import random
import yaml
import traceback
import subprocess
from mainwindow import Ui_MainWindow
import cv2
import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication, QThread, pyqtSignal, Qt
from PyQt5.QtGui import *
from ultralytics.utils.files import increment_path
from pathlib import Path
import time
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
VIDEO_SUFFIXES = {'.mp4', '.flv', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.ts', '.mpeg', '.mpg'}
MEDIA_FILE_FILTER = (
    '媒体文件('
    '*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp '
    '*.mp4 *.flv *.avi *.mov *.mkv *.wmv *.m4v *.ts *.mpeg *.mpg'
    ')'
)


def cvimg_to_qtimg(cvimg):
    """
    将 OpenCV 图像转换为 PyQt 图像
    """
    if cvimg is None:
        return QImage()

    if len(cvimg.shape) == 2:
        rgb_img = cv2.cvtColor(cvimg, cv2.COLOR_GRAY2RGB)
        qimg = QImage(
            rgb_img.data,
            rgb_img.shape[1],
            rgb_img.shape[0],
            rgb_img.strides[0],
            QImage.Format_RGB888
        )
        return qimg.copy()

    if cvimg.shape[2] == 4:
        rgba_img = cv2.cvtColor(cvimg, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(
            rgba_img.data,
            rgba_img.shape[1],
            rgba_img.shape[0],
            rgba_img.strides[0],
            QImage.Format_RGBA8888
        )
        return qimg.copy()

    rgb_img = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    qimg = QImage(
        rgb_img.data,
        rgb_img.shape[1],
        rgb_img.shape[0],
        rgb_img.strides[0],
        QImage.Format_RGB888
    )
    return qimg.copy()


class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_text_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    
    def __init__(self, model, source, save_result=False, save_dir=None, parent=None):
        super(DetectionThread, self).__init__(parent)
        self.model = model
        self.source = source
        self.save_result = save_result
        self.save_dir = save_dir
        self.output_path = ''
        self.running = True
        self.paused = False
        self.category = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def run(self):
        cap = cv2.VideoCapture(self.source)

        # Video writer setup
        vid_writer = None
        try:
            if not cap.isOpened():
                self.running = False
                self.update_text_signal.emit("无法打开视频源。")
                return

            source_suffix = Path(str(self.source)).suffix.lower() if not isinstance(self.source, int) else ''
            is_image_source = source_suffix in IMAGE_SUFFIXES
            if self.save_result and (isinstance(self.source, int) or not is_image_source):
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if fps == 0:
                    fps = 30
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                filename = os.path.basename(str(self.source)) if self.source != 0 else 'camera'
                if self.source == 0:
                    filename = 'camera'

                # Ensure save_dir exists
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir, exist_ok=True)

                save_path = str(Path(str(self.save_dir) + '/' + str(filename)).with_suffix('.mp4'))
                self.output_path = save_path
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)

            t1 = time.time()

            while self.running:
                if self.paused:
                    self.msleep(100)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                # Detection
                # Use verbose=False to reduce console spam
                results = self.model(frame, verbose=False)
                boxes = results[0].boxes

                # Plot
                plot_args = dict(line_width=None, boxes=True, conf=True, labels=True)
                plotted_img = results[0].plot(**plot_args)

                # Convert to QImage
                qt_img = cvimg_to_qtimg(plotted_img.copy())
                self.change_pixmap_signal.emit(qt_img)

                # Text output
                t2 = time.time()
                t = t2 - t1
                t1 = t2

                pred = ''
                for i in range(len(boxes.data)):
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    if cls_id < len(self.category):
                        boxes_cls = self.category[cls_id]
                        pred += f'坐标 {boxes.xyxy[i].cpu().numpy()}：{boxes_cls}\n\n'
                pred += f"用时：{t:.4f}s"
                self.update_text_signal.emit(pred)

                # Save video
                if vid_writer:
                    vid_writer.write(plotted_img)
        except Exception:
            self.update_text_signal.emit(f"检测线程异常：\n{traceback.format_exc()}")
        finally:
            if vid_writer:
                vid_writer.release()
            cap.release()
            self.finished_signal.emit(self.output_path)
    
    def stop(self):
        self.running = False

    def pause_toggle(self):
        self.paused = not self.paused


class ModelLoadThread(QThread):
    loaded_signal = pyqtSignal(object, str)
    error_signal = pyqtSignal(str)

    def __init__(self, weight_path, parent=None):
        super(ModelLoadThread, self).__init__(parent)
        self.weight_path = weight_path

    def run(self):
        try:
            model = YOLO(str(self.weight_path))
            self.loaded_signal.emit(model, str(self.weight_path))
        except Exception as e:
            self.error_signal.emit(str(e))


class ImageDetectionThread(QThread):
    finished_signal = pyqtSignal(QImage, str, str)
    error_signal = pyqtSignal(str)

    def __init__(self, model, image_path, save_result, root_path, category, parent=None):
        super(ImageDetectionThread, self).__init__(parent)
        self.model = model
        self.image_path = image_path
        self.save_result = save_result
        self.root_path = root_path
        self.category = category

    def run(self):
        try:
            img0 = cv2.imread(self.image_path)
            if img0 is None:
                raise RuntimeError("无法读取图片文件")

            results = self.model(img0, verbose=False)
            boxes = results[0].boxes
            plotted_img = results[0].plot(line_width=None, boxes=True, conf=True, labels=True)
            result_qt = cvimg_to_qtimg(plotted_img.copy())

            save_path = ""
            if self.save_result:
                filename = os.path.basename(self.image_path)
                save_dir = increment_path(Path(self.root_path / 'runs' / 'detect') / 'predict', exist_ok=False, mkdir=True)
                os.makedirs(save_dir, exist_ok=True)
                save_path = str(Path(str(save_dir) + '/' + str(filename)))
                cv2.imwrite(save_path, plotted_img)

            pred = ''
            for i in range(len(boxes.data)):
                cls_id = int(boxes.cls[i].cpu().numpy())
                if cls_id < len(self.category):
                    boxes_cls = self.category[cls_id]
                    pred += '坐标 ' + str(boxes.xyxy[i].cpu().numpy()) + '：' + str(boxes_cls) + '\n\n'

            self.finished_signal.emit(result_qt, pred, save_path)
        except Exception:
            self.error_signal.emit(traceback.format_exc())


class TrainThread(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, model_name, yaml_path, epoch, batch, workers, save, optimizer, root_path, parent=None):
        super(TrainThread, self).__init__(parent)
        self.model_name = model_name
        self.yaml_path = yaml_path
        self.epoch = epoch
        self.batch = batch
        self.workers = workers
        self.save = save
        self.optimizer = optimizer
        self.root_path = root_path
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True
        self.log_signal.emit('已请求停止训练，将在当前 epoch 结束后中止。')

    def _on_fit_epoch_end(self, trainer):
        try:
            current_epoch = int(getattr(trainer, "epoch", -1)) + 1
            total_epoch = int(getattr(trainer, "epochs", self.epoch))
            loss_items = getattr(trainer, "loss_items", None)
            if loss_items is not None and len(loss_items) > 0:
                if hasattr(loss_items, "tolist"):
                    loss_text = ", ".join([f"{float(x):.4f}" for x in loss_items.tolist()])
                else:
                    loss_text = str(loss_items)
                self.log_signal.emit(f"[训练进度] epoch {current_epoch}/{total_epoch}, loss: {loss_text}")
            else:
                self.log_signal.emit(f"[训练进度] epoch {current_epoch}/{total_epoch}")
        except Exception:
            self.log_signal.emit("[训练进度] epoch 完成")

        if self.stop_requested:
            trainer.stop = True

    def run(self):
        try:
            self.log_signal.emit(f"开始训练：model={self.model_name}, data={self.yaml_path}")
            model = YOLO(self.model_name)
            model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
            model.train(
                data=self.yaml_path,
                epochs=self.epoch,
                batch=self.batch,
                workers=self.workers,
                save=self.save,
                optimizer=self.optimizer,
                project=str(self.root_path / "runs"),
                name="train/exp"
            )

            if self.stop_requested:
                self.log_signal.emit("训练已中止，跳过验证。")
                self.done_signal.emit()
                return

            self.log_signal.emit("训练完成，开始验证...")
            model.val(project=str(self.root_path / "runs"), name="val/exp")
            self.log_signal.emit("验证完成。")
            self.done_signal.emit()
        except Exception:
            self.error_signal.emit(traceback.format_exc())


class UiMain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(UiMain, self).__init__(parent)
        self.model = None  # 模型
        self.setupUi(self)

        self.OpenButton.setVisible(False)  # 导入图片/视频
        self.OpenButton.clicked.connect(self.loadimage)

        self.EndButton.clicked.connect(self.close)  # 退出程序

        self.fname = None  # 文件名
        self.weight_path = 'yolov8n.pt'

        self.ReButton.setVisible(False)  # 重新检测
        self.ReButton.clicked.connect(self.rework)

        self.starButton.clicked.connect(self.model_star)  # 初始化模型

        self.stopButton.setVisible(False)  # 停止检测
        self.stopButton.clicked.connect(self.stop_detection)

        self.PauseButton.setVisible(False)  # 暂停检测
        self.PauseButton.clicked.connect(self.pause_detection)

        self.ContinueButton.setVisible(False)  # 继续检测
        self.ContinueButton.clicked.connect(self.continue_detection)

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

        self.predefined_classes = None
        self.yaml_path = None
        self.category = None
        self.label_directory = None
        self.data_to_split = None
        
        self.ROOT = Path(__file__).resolve().parents[0]
        
        # Thread
        self.det_thread = None
        self.model_thread = None
        self.img_thread = None
        self.train_thread = None
        self.exit_pending = False

    def weight(self):
        self.weight_path, _ = QFileDialog.getOpenFileName(self, '请选择权重文件', '.', '权重文件(*.pt)')

    def model_star(self):
        if not self.weight_path:
            self.weight_path = 'yolov8n.pt'
        if self.model_thread and self.model_thread.isRunning():
            self.statusbar.showMessage('模型加载中，请稍候...', 3000)
            return

        self.starButton.setEnabled(False)
        self.weightButton.setEnabled(False)
        self.textEdit.setText(f"正在初始化模型：{self.weight_path}")

        self.model_thread = ModelLoadThread(self.weight_path)
        self.model_thread.loaded_signal.connect(self.on_model_loaded)
        self.model_thread.error_signal.connect(self.on_model_load_error)
        self.model_thread.start()
        return

    def on_model_loaded(self, model, weight_path):
        self.model = model
        self.weightButton.setVisible(False)
        self.OpenButton.setVisible(True)
        self.cameraButton.setVisible(True)
        self.textEdit.setText(f"模型初始化成功：{weight_path}")
        self.starButton.setVisible(False)
        self.category = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.starButton.setEnabled(True)
        self.weightButton.setEnabled(True)
        self._attempt_quit_if_ready()

    def on_model_load_error(self, error):
        self.starButton.setEnabled(True)
        self.weightButton.setEnabled(True)
        QMessageBox.warning(self, "错误", f"模型初始化失败：{error}")
        self._attempt_quit_if_ready()

    def camera(self):
        self.fname = 0
        self.statusbar.showMessage('已启动摄像头', 10000)
        self.work()

    def loadimage(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, '请选择图片或视频', '.', MEDIA_FILE_FILTER)
        if self.fname:
            self.statusbar.showMessage(f'已成功打开文件：{self.fname}', 10000)
            self.work()
        else:
            self.textEdit.setText("打开文件失败")

    def rework(self):
        result = QMessageBox.question(
            self,
            "确认",
            "是否重新检测？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if result == QMessageBox.Yes:
            self.work()
            self.statusbar.showMessage(f'已重新检测：{self.fname}', 10000)

    def work(self):
        if self.fname is None:
            return

        # Stop existing thread if running
        if self.det_thread and self.det_thread.isRunning():
            self.det_thread.stop()
            self.statusbar.showMessage('正在停止上一次检测，请稍后重试。', 2000)
            return

        # Check for image vs video
        is_video = False
        if self.fname == 0:
            is_video = True
        elif Path(str(self.fname)).suffix.lower() in VIDEO_SUFFIXES:
            is_video = True
            
        if is_video:
            self.setup_video_ui_state()
            
            # Prepare save directory if needed
            save_dir = None
            if self.checkBox.isChecked():
                save_dir = increment_path(Path(self.ROOT / 'runs' / 'detect') / 'predict', exist_ok=False, mkdir=True)
            
            self.det_thread = DetectionThread(self.model, self.fname, self.checkBox.isChecked(), save_dir)
            self.det_thread.change_pixmap_signal.connect(self.update_image)
            self.det_thread.update_text_signal.connect(self.update_text)
            self.det_thread.finished_signal.connect(self.on_detection_finished)
            self.det_thread.start()
            
        else:
            # Single image
            self.detect_single_image_async()
            self.ReButton.setVisible(True)

    def setup_video_ui_state(self):
        self.cameraButton.setVisible(False)
        self.ReButton.setVisible(False)
        self.EndButton.setVisible(False)
        self.OpenButton.setVisible(False)
        self.stopButton.setEnabled(True)
        self.PauseButton.setEnabled(True)
        self.ContinueButton.setEnabled(True)
        self.ContinueButton.setVisible(False)
        self.PauseButton.setVisible(True)
        self.stopButton.setVisible(True)

    def detect_single_image_async(self):
        if self.img_thread and self.img_thread.isRunning():
            self.statusbar.showMessage('图片检测中，请稍候...', 3000)
            return
        self.textEdit.setText('正在进行图片检测...')
        self.img_thread = ImageDetectionThread(self.model, self.fname, self.checkBox.isChecked(), self.ROOT, self.category)
        self.img_thread.finished_signal.connect(self.on_single_image_finished)
        self.img_thread.error_signal.connect(self.on_single_image_error)
        self.img_thread.start()

    def on_single_image_finished(self, result_qt, pred, save_path):
        self.update_image(result_qt)
        self.textEdit.setText(pred)
        if save_path:
            self.statusbar.showMessage(f'检测结果已保存：{save_path}', 5000)
        self._attempt_quit_if_ready()

    def on_single_image_error(self, error):
        QMessageBox.warning(self, "错误", f"图片检测失败：\n{error}")
        self._attempt_quit_if_ready()

    def update_image(self, qt_img):
        if not qt_img.isNull():
            jpg = QtGui.QPixmap(qt_img).scaled(self.Imglabel.width(), self.Imglabel.height(), Qt.KeepAspectRatio)
            self.Imglabel.setPixmap(jpg)

    def update_text(self, text):
        self.textEdit.setText(text)

    def stop_detection(self):
        if self.det_thread and self.det_thread.isRunning():
            self.det_thread.stop()
            self.stopButton.setEnabled(False)
            self.PauseButton.setEnabled(False)
            self.ContinueButton.setEnabled(False)
            self.statusbar.showMessage('正在停止检测，请稍候...', 3000)
            return
        self.reset_ui_state()

    def pause_detection(self):
        if self.det_thread:
            self.det_thread.pause_toggle()
        self.PauseButton.setVisible(False)
        self.ContinueButton.setVisible(True)

    def continue_detection(self):
        if self.det_thread:
            self.det_thread.pause_toggle()
        self.PauseButton.setVisible(True)
        self.ContinueButton.setVisible(False)
        
    def on_detection_finished(self, save_path):
        if save_path:
            self.statusbar.showMessage(f'检测结果已保存：{save_path}', 5000)
        self.reset_ui_state()
        self._attempt_quit_if_ready()
        
    def reset_ui_state(self):
        self.stopButton.setEnabled(True)
        self.PauseButton.setEnabled(True)
        self.ContinueButton.setEnabled(True)
        self.stopButton.setVisible(False)
        self.PauseButton.setVisible(False)
        self.ContinueButton.setVisible(False)
        self.OpenButton.setVisible(True)
        self.EndButton.setVisible(True)
        self.cameraButton.setVisible(True)
        if self.fname != 0:
            self.ReButton.setVisible(True)

    def close(self):
        if self.exit_pending:
            self._attempt_quit_if_ready()
            return

        result = QMessageBox.question(
            self,
            "确认",
            "是否退出程序？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if result == QMessageBox.Yes:
            self.exit_pending = True
            if self.det_thread and self.det_thread.isRunning():
                self.det_thread.stop()
            if self.train_thread and self.train_thread.isRunning():
                self.train_thread.request_stop()
            self._attempt_quit_if_ready()

    def label_f(self):
        self.label_directory = QFileDialog.getExistingDirectory(self, "选择文件夹", "C:/")
        if self.label_directory:
            self.lineEdit.setText(self.label_directory)

    def safe(self):
        if self.label_directory:
            classes = self.classEdit.toPlainText().strip()
            class_list = [line.strip() for line in classes.splitlines() if line.strip()]
            num = len(class_list)
            classes_num = self.comboBox.currentText()
            if num != int(classes_num) or num == 0:
                QMessageBox.question(self, "提示", "类别数量不匹配。", QMessageBox.Ok)
            else:
                target_file = os.path.join(self.label_directory, '../predefined_classes.txt')
                target_file = os.path.abspath(target_file)
                try:
                    with open(target_file, 'w') as file:
                        file.write(classes)
                    self.predefined_classes = target_file
                    self.statusbar.showMessage('已生成 predefined_classes.txt', 3000)
                except Exception as e:
                     QMessageBox.warning(self, "错误", f"写入文件失败：{e}")
        else:
            QMessageBox.question(self, "提示", "请先选择图片目录。", QMessageBox.Ok)

    def start_label(self):
        if self.label_directory:
            labelimg_dir = os.path.join(self.ROOT, 'labelimg')
            cmd = [sys.executable, "labelimg.py", self.label_directory]
            if self.predefined_classes:
                cmd.append(self.predefined_classes)
            else:
                maybe_classes = os.path.abspath(os.path.join(self.label_directory, '../predefined_classes.txt'))
                if os.path.exists(maybe_classes):
                    cmd.append(maybe_classes)

            try:
                subprocess.Popen(cmd, cwd=labelimg_dir)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"启动 LabelImg 失败：{e}")
                
        else:
            QMessageBox.question(self, "提示", "请先选择图片目录。", QMessageBox.Ok)

    def pre_split(self):
        self.data_to_split = QFileDialog.getExistingDirectory(self, "选择文件夹", "C:/")
        if self.data_to_split:
            self.lineEdit_2.setText(self.data_to_split)

    def split_data(self):
        if self.data_to_split:
            images_dir = self.data_to_split
            if not str(images_dir).endswith('images'):
                QMessageBox.question(self, "提示", "图片目录名需为 images 才能正常训练。", QMessageBox.Ok)
                return
            
            try:
                image_suffixes = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
                data_list = [
                    name for name in os.listdir(images_dir)
                    if os.path.isfile(os.path.join(images_dir, name)) and name.lower().endswith(image_suffixes)
                ]
                num = len(data_list)
                if num < 2:
                    QMessageBox.question(self, "提示", "图片数量至少为 2 张才能划分训练/验证集。", QMessageBox.Ok)
                    return
                rate = 0.1
                val_count = max(1, int(num * rate))
                val = random.sample(data_list, val_count)
                train = list(set(data_list) - set(val))
                
                parent_dir = os.path.dirname(images_dir)
                train_txt = os.path.join(parent_dir, 'train.txt')
                val_txt = os.path.join(parent_dir, 'val.txt')
                
                with open(train_txt, 'w') as train_file:
                    for i in train:
                        train_file.write(os.path.join(images_dir, i) + '\n')
                with open(val_txt, 'w') as val_file:
                    for i in val:
                        val_file.write(os.path.join(images_dir, i) + '\n')
                        
                self.statusbar.showMessage('已生成 train.txt 和 val.txt。', 5000)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"划分数据集失败：{e}")
        else:
            QMessageBox.question(self, "提示", "请先选择图片目录。", QMessageBox.Ok)

    def create_yaml(self):
        if self.data_to_split:
            parent_dir = os.path.dirname(self.data_to_split)
            predefined_classes_path = os.path.join(parent_dir, 'predefined_classes.txt')
            train_txt_path = os.path.join(parent_dir, 'train.txt')
            val_txt_path = os.path.join(parent_dir, 'val.txt')
            
            if os.path.exists(predefined_classes_path) and os.path.exists(train_txt_path) and \
                    os.path.exists(val_txt_path):
                num = 0
                class_dic = {}
                with open(predefined_classes_path, 'r') as file:
                    classes = file.readlines()
                for i in range(len(classes)):
                    if classes[i].strip():
                        class_dic[num] = classes[i].strip()
                        num += 1
                
                desired_caps = {
                    'train': os.path.abspath(train_txt_path),
                    'val': os.path.abspath(val_txt_path),
                    'names': class_dic,
                    'nc': num
                }
                
                self.yaml_path = os.path.join(self.ROOT, 'tmp.yaml')
                with open(self.yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(desired_caps, f)
                self.statusbar.showMessage(f'已创建 YAML：{os.path.abspath(self.yaml_path)}', 5000)
            else:
                QMessageBox.question(self, "提示", "数据集目录缺少所需文件。", QMessageBox.Ok)
        else:
            QMessageBox.question(self, "提示", "请先选择训练数据集。", QMessageBox.Ok)

    def epoch_f(self):
        self.epoch = self.epoch_Slider.value()
        self.epoch_label.setText(f'训练轮数：{self.epoch}')

    def batch_f(self):
        self.batch = self.batch_Slider.value()
        self.batch_label.setText(f'批大小：{self.batch}')

    def workers_f(self):
        self.workers = self.workers_Slider.value()
        self.workers_label.setText(f'数据加载线程数：{self.workers}')

    def train(self):
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.request_stop()
            self.train_Button.setText("停止中...")
            return

        save = self.checkBox_2.isChecked()
        model_name = str(self.model_comboBox.currentText())
        
        if self.yaml_comboBox.currentIndex() == 0 and not self.yaml_path:
            QMessageBox.question(self, "提示", "请先创建或选择 YAML 文件。", QMessageBox.Ok)
            return 0
        elif self.yaml_comboBox.currentIndex() == 0 and self.yaml_path:
            yaml_path = self.yaml_path
        else:
            yaml_path = str(self.yaml_comboBox.currentText())
        yaml_path = os.path.abspath(yaml_path)
        if not os.path.isfile(yaml_path):
            QMessageBox.question(self, "提示", f"YAML 文件不存在：{yaml_path}", QMessageBox.Ok)
            return

        self.train_thread = TrainThread(
            model_name=model_name,
            yaml_path=yaml_path,
            epoch=self.epoch,
            batch=self.batch,
            workers=self.workers,
            save=save,
            optimizer=self.optimizer_comboBox.currentText(),
            root_path=self.ROOT
        )
        self.train_thread.log_signal.connect(self.append_train_log)
        self.train_thread.done_signal.connect(self.on_train_done)
        self.train_thread.error_signal.connect(self.on_train_error)
        self.train_thread.start()
        self.train_Button.setText("停止训练")
        self.statusbar.showMessage("训练已启动。", 3000)
        return


    def append_train_log(self, text):
        self.textEdit.append(text)

    def on_train_done(self):
        self.train_Button.setText("训练")
        if not self.exit_pending:
            QMessageBox.question(self, "提示", "训练完成。", QMessageBox.Ok)
        self._attempt_quit_if_ready()

    def on_train_error(self, error):
        self.train_Button.setText("训练")
        QMessageBox.warning(self, "错误", f"训练失败：\n{error}")
        self._attempt_quit_if_ready()

    def _has_running_threads(self):
        threads = (self.det_thread, self.img_thread, self.model_thread, self.train_thread)
        return any(thread is not None and thread.isRunning() for thread in threads)

    def _attempt_quit_if_ready(self):
        if not self.exit_pending:
            return
        if self._has_running_threads():
            self.statusbar.showMessage("正在等待线程安全退出...", 3000)
            return
        self._quit_app()

    def _quit_app(self):
        app_ = QApplication.instance()
        app_.quit()

    def closeEvent(self, event):
        if not self.exit_pending:
            event.ignore()
            self.close()
            return
        if self._has_running_threads():
            event.ignore()
            return
        event.accept()


if __name__ == '__main__':
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    ui = UiMain()
    ui.show()
    sys.exit(app.exec_())


