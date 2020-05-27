from imaging import Session, Intrinsic, MOVIE_EXPORT, overlay
import sys
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
from skimage.filters import gaussian as gauss_filt
from skimage.io import imsave
from pathlib import Path
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler


class ViewerIntrinsic(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Read color map from here : http://www.kennethmoreland.com/color-advice/
        self.cl = np.loadtxt('extended-black-body-table-byte-0256.csv', delimiter=',', skiprows=1)
        self.cmap = [QtGui.qRgb(*x[1:]) for x in self.cl]
        self.cl = np.vstack((np.ones(self.cl.shape[0]), self.cl[:, 1:].transpose())).transpose()
        self.c_data = np.array([])
        self._c_slice = 0
        self.S: Optional[Session] = None
        # File model
        home = str(Path.home())
        self.file_model = QtWidgets.QFileSystemModel()
        self.file_model.setFilter(QtCore.QDir.AllDirs | QtCore.QDir.Files | QtCore.QDir.NoDotAndDotDot )
        self.file_model.setNameFilters(["*.h5"])
        self.file_model.setRootPath(home)

        self.main_widget = QtWidgets.QWidget(self)
        # Layouts
        self.h_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.h_layout_cble = QtWidgets.QHBoxLayout()
        self.h_layout_btn = QtWidgets.QHBoxLayout()
        self.v_layout_left = QtWidgets.QVBoxLayout()
        self.v_layout_right = QtWidgets.QVBoxLayout()
        self.h_layout_analysis = QtWidgets.QHBoxLayout()
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_wdg = QtWidgets.QWidget()
        left_wdg.setLayout(self.v_layout_left)
        left_scroll.setWidget(left_wdg)
        # Widgets
        self.tree = QtWidgets.QTreeView(self)
        self.tree.setModel(self.file_model)
        self.tree.hideColumn(1)
        self.tree.hideColumn(2)
        self.tree.hideColumn(3)
        self.tree.expanded.connect(self.expanded)
        self.tree.selectionModel().currentChanged.connect(self.select_file)
        self.analysis_btn = QtWidgets.QPushButton('&Analysis')
        self.analysis_btn.clicked.connect(self.analyze)
        self.movie_btn = QtWidgets.QPushButton('&Movie')
        self.movie_btn.clicked.connect(self.export_movie)
        self.resp_btn = QtWidgets.QPushButton('&Save response')
        self.resp_btn.clicked.connect(self.export_resp)
        self.tc_btn = QtWidgets.QPushButton('Save &time course')
        self.tc_btn.clicked.connect(self.export_tc)
        self.excel_btn = QtWidgets.QPushButton('Excel export')
        self.excel_btn.clicked.connect(self.excel_export)
        self.max_slider = LabeledSlider('Maximum value')
        self.max_slider.setEnabled(False)
        self.max_slider.setSingleStep(1)
        self.max_slider.valueChanged.connect(self.max_df_changed)
        self.slice_slider = LabeledSlider('Current frame')
        self.slice_slider.setEnabled(False)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.valueChanged.connect(self.slice_changed)
        self.data_cb = QtWidgets.QComboBox(self)
        self.data_cb.currentIndexChanged.connect(self.data_changed)
        self.data_cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.comment_le = QtWidgets.QLineEdit()
        self.comment_le.setEnabled(False)
        self.comment_le.editingFinished.connect(self.commenting)
        # PyQtGraph
        self.win = pg.GraphicsLayoutWidget(self)
        self.plot_anat = self.win.addPlot(row=0, col=0)
        self.plot_resp = self.win.addPlot(row=1, col=0)
        self.resp_item = pg.ImageItem()
        self.anat_item = pg.ImageItem()
        self.plot_resp.addItem(self.resp_item)
        self.plot_anat.addItem(self.anat_item)
        roi_pen = pg.mkPen(color='y', width=2)
        self.roi = pg.ROI([0, 0], [100, 100], pen=roi_pen)
        self.roi_anat = pg.ROI([0, 0], [100, 100], movable=False, pen=roi_pen)
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.roi.sigRegionChanged.connect(self.roi_moved)
        self.plot_resp.addItem(self.roi)
        self.plot_anat.addItem(self.roi_anat)
        self.roi.setZValue(10)
        self.plot_widget = self.win.addPlot(row=2, col=0)
        self.roi_plot = pg.PlotDataItem()
        self.plot_widget.addItem(self.roi_plot)
        # Adding widgets to layouts
        self.v_layout_right.addWidget(self.win)
        self.v_layout_left.addWidget(self.tree)
        self.h_layout_analysis.addSpacerItem(QtWidgets.QSpacerItem(300, 1, QtWidgets.QSizePolicy.Expanding))
        self.h_layout_analysis.addWidget(self.analysis_btn)
        self.v_layout_left.addLayout(self.h_layout_analysis)
        self.v_layout_left.addWidget(self.max_slider)
        self.h_layout_btn.addWidget(self.movie_btn)
        self.h_layout_btn.addWidget(self.resp_btn)
        self.h_layout_btn.addWidget(self.tc_btn)
        self.h_layout_btn.addWidget(self.excel_btn)
        self.h_layout_cble.addWidget(self.data_cb)
        self.h_layout_cble.addWidget(self.comment_le)
        self.v_layout_left.addLayout(self.h_layout_cble)
        self.v_layout_left.addWidget(self.slice_slider)
        self.v_layout_left.addLayout(self.h_layout_btn)
        # self.h_layout.addLayout(self.v_layout_left)
        self.h_layout.addWidget(left_scroll)
        self.h_layout.addLayout(self.v_layout_right)
        # Sizing
        self.setMinimumSize(600, 800)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # self.tree.setMinimumSize(400, 800)
        self.setCentralWidget(self.main_widget)
        self.setWindowTitle('Flavo imaging explorer')
        self.c_path = None
        self.an_th = None
        self.resp_btn.setEnabled(False)
        self.tc_btn.setEnabled(False)
        self.movie_btn.setEnabled(False)
        self.excel_btn.setEnabled(False)
        # Logging
        self._logger = logging.getLogger('Intrinsiclog')
        self._logger.setLevel(logging.DEBUG)
        self._log_handler = RotatingFileHandler('Intrinsic.log', maxBytes=int(1e6), backupCount=1)
        formatter = logging.Formatter(
            '%(asctime)s :: %(filename)s :: %(funcName)s :: line %(lineno)d :: %(levelname)s :: %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S')
        self._log_handler.setFormatter(formatter)
        self._logger.addHandler(self._log_handler)
        sys.excepthook = handle_exception

        self.statusBar().showMessage('Ready!', 1500)

    def analyze(self):
        if self.S is not None:
            self.S.close()
            self.S = None
        # c_item = self.tree.model().itemData(self.tree.currentIndex())[0]
        c_path = Path(self.file_model.filePath(self.tree.currentIndex()))
        if not c_path.is_dir():
            return
        # Get all children directories
        children = [p for p in c_path.iterdir() if p.is_dir()]
        # Check for png in children directories
        n_png = [len(list(p.glob('*.png'))) for p in children]
        if any([n > 10 for n in n_png]) and self.an_th is None:
            # Has good chances to be a proper recording folder
            self.statusBar().showMessage('Analysis has started')
            self.an_th = AnalysisThread(self, c_path, binning=3)
            self.an_th.finished.connect(self.finished_analysis)
            self.an_th.start()

    def finished_analysis(self):
        self.statusBar().showMessage('Analysis is done')
        self.an_th.quit()
        self.an_th.wait()

        self.an_th = None

    def closeEvent(self, a0: QtGui.QCloseEvent):
        if self.S is not None:
            self.S.file.close()
            print("File closed")
        a0.accept()

    def excel_export(self):
        if self.S is None:
            return
        ((xs, xe), (ys, ye)), _ = self.roi.getArraySlice(self.S.avg_stack, self.resp_item,
                                                         returnSlice=False)
        save_path = self.S.export_tc_to_csv(xs, xe, ys, ye)
        self.S.export_resp_prm()
        self.statusBar().showMessage(f'Data saved to {save_path}', 5000)

    def export_tc(self):
        if self.S is None:
            return
        ((xs, xe), (ys, ye)), _ = self.roi.getArraySlice(self.S.avg_stack, self.resp_item,
                                                         returnSlice=False)
        self.S.export_timecourse(xs, xe, ys, ye)

    def export_resp(self):
        if self.S is None:
            return
        self.S.export_response()
        im_over = overlay(self.S.anat, self.S.norm_stack, self.S.resp_map)
        imsave(self.S.path.parent / f'overlay_{self.S.path.name}.png', im_over)
        # DONE : Save the overlay

    def export_movie(self):
        if self.S is None:
            return
        self.S.export_movie()

    def commenting(self):
        if self.S is None:
            return
        self.S.comment = self.comment_le.text()

    def max_df_changed(self, value):
        self.resp_item.setImage(self.norm_pic(self.S.resp))

    def slice_changed(self, value):
        self.c_slice = value

    def norm_pic(self, pic):
        b_pic = pic.copy()
        b_pic = np.clip(b_pic, None, self.max_slider.value()/1000)
        b_pic = gauss_filt(b_pic, 3)
        divider = (b_pic.max() - b_pic.min())
        if divider == 0:
            divider = 1
        b_pic = 255 * (b_pic - b_pic.min()) / divider
        b_pic = np.uint8(b_pic)
        return b_pic

    def select_file(self, current):
        c_item = self.tree.model().itemData(current)[0]
        if c_item.endswith('.h5'):
            self.data_cb.clear()
            c_path = self.file_model.filePath(current)
            if self.S is not None:
                self.S.close()
            self.S = Session(c_path)
            self.setWindowTitle(f'Flavo imaging explorer - {Path(c_path).name}')
            self.max_slider.setEnabled(True)
            self.max_slider.setMinimum(0)
            self.max_slider.setMaximum(int(1000*self.S.resp.max()))
            self.max_slider.setValue(50)
            self.c_path = c_path
            self.roi_moved()
            self.data_cb.addItem('Average stack', 'avg_stack')
            self.data_cb.addItem('Normalized stack', 'norm_stack')
            self.data_cb.addItem('Max projection', 'resp')
            self.comment_le.setEnabled(True)
            self.resp_btn.setEnabled(True)
            self.tc_btn.setEnabled(True)
            self.movie_btn.setEnabled(MOVIE_EXPORT)
            self.excel_btn.setEnabled(True)
            self.data_cb.setCurrentIndex(2)
            self.anat_item.setImage(self.S.anat)
            self.comment_le.setText(self.S.comment)

    def data_changed(self, ix: int):
        if ix == -1 or self.S is None:
            return
        self.roi_moved()
        data = self.S.__getattribute__(self.data_cb.itemData(ix))
        if len(data.shape) == 3:
            self.slice_slider.setEnabled(True)
            self.slice_slider.setRange(0, data.shape[2]-1)
        else:
            self.slice_slider.setEnabled(False)
        self.c_data = data
        self.c_slice = self._c_slice
        self.slice_slider.setValue(0)

    @property
    def c_slice(self):
        return self._c_slice

    @c_slice.setter
    def c_slice(self, value):
        if self.S is None:
            return
        self._c_slice = value
        if len(self.c_data.shape) == 3:
            data = self.c_data[..., value]
        else:
            data = self.c_data
        self.resp_item.setImage(self.norm_pic(data))

    def expanded(self, index):
        self.tree.resizeColumnToContents(0)

    def roi_moved(self):
        if self.S is None:
            return
        self.roi_anat.setState(self.roi.getState())
        if self.c_path is None:
            return
        ((xs, xe), (ys, ye)), _ = self.roi.getArraySlice(self.S.avg_stack, self.resp_item,
                                                         returnSlice=False)
        if self.data_cb.currentData() != 'avg_stack':
            slice_data = self.S.norm_stack[xs:xe, ys:ye, :].mean((0,1))
        else:
            slice_data = self.S.avg_stack[xs:xe, ys:ye, :].mean((0,1))
        bi = ((slice_data - slice_data.mean()) / slice_data.std()) < -2
        slice_data[bi] = np.nan
        self.roi_plot.setData(x=np.arange(len(slice_data)), y=slice_data)


class AnalysisThread(QtCore.QThread):
    def __init__(self, parent, datapath, **kwargs) -> None:
        """

        Parameters
        ----------
        datapath : path
        """
        super().__init__(parent)
        self.parent = parent
        self.datapath = datapath
        self.kwargs = kwargs

    def run(self) -> None:
        I = Intrinsic(self.datapath, **self.kwargs)
        I.save_analysis()


class ImLab(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self._pix = None

    def setPixmap(self, a0: QtGui.QPixmap) -> None:
        if self._pix is None:
            self._pix = a0
        super().setPixmap(a0)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        super().paintEvent(a0)
        if self._pix is None:
            return
        im_height, im_width = self._pix.height(), self._pix.width()
        ratio = im_width / im_height
        scale = min(self.width() / im_width, self.height() / im_height)
        new_width = im_width * scale
        new_height = new_width / ratio
        pix = self._pix.scaled(new_height, new_width)
        self.setPixmap(pix)


class LabeledSlider(QtWidgets.QWidget):

    def __init__(self, title='') -> None:
        super(LabeledSlider, self).__init__()
        self._title = title
        self.title_label = QtWidgets.QLabel(self.title)
        self.value_label = QtWidgets.QLabel('')
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.value_changed)
        self.v_layout = QtWidgets.QVBoxLayout()
        self.h_layout = QtWidgets.QHBoxLayout()
        self.h_layout.addWidget(self.slider)
        self.h_layout.addWidget(self.value_label)
        self.v_layout.addWidget(self.title_label)
        self.v_layout.addLayout(self.h_layout)
        self.setLayout(self.v_layout)
        self.valueChanged = self.slider.valueChanged
        self.setMinimum = self.slider.setMinimum
        self.setMaximum = self.slider.setMaximum
        self.setValue = self.slider.setValue
        self.value = self.slider.value
        self.setSingleStep = self.slider.setSingleStep
        self.setEnabled = self.slider.setEnabled
        self.setRange = self.slider.setRange

    def value_changed(self, value: int) -> None:
        # self.slider.valueChanged(value)
        self.value_label.setText(str(value))

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self.title_label.setText(value)


def normalize_stack(stack, n_baseline=30):
    start = 10
    # Global average
    y = np.nanmean(stack, (0, 1))[start:n_baseline]
    y_min, y_max = y.min(), y.max()
    # Exponential fit during baseline
    t = np.arange(start, n_baseline)
    z = 1 + (y - y_min) / (y_max - y_min)
    p = np.polyfit(t, np.log(z), 1)
    # Modeled decay
    full_t = np.arange(stack.shape[2])
    decay = np.exp(p[1]) * np.exp(full_t * p[0])
    # Renormalized
    decay = (decay - 1) * (y_max - y_min) + y_min
    norm_stack = stack - decay
    norm_stack = gauss_filt(norm_stack, 3, multichannel=False)
    return norm_stack


def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions and print in logger."""
    logger = logging.getLogger('Intrinsiclog')
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    window = ViewerIntrinsic()
    window.show()
    sys.exit(qApp.exec_())
