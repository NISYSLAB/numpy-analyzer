# libraries needed for main python file
# the file we run for the app to work
import sys
import numpy as np
import os
import time
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import plotly.express as px
import plotly.graph_objects as go
from viewer_GUI import Ui_MainWindow
import viewer.visualization as viz
import viewer.load_data as load_data
import viewer.generate_signal as generate_signal
import viewer.filters as filter
from viewer.segment_data import separate_data_chs
from scipy import signal

# class definition for application window components like the ui
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.scroll_bar = QScrollBar(self)
        self.ui.FilesListWidget.setVerticalScrollBar(self.scroll_bar)
        self.ui.FilesListWidget.setHorizontalScrollBar(self.scroll_bar)
        self.signal_plot = QtWebEngineWidgets.QWebEngineView(self)
        self.signal_prop_plot = QtWebEngineWidgets.QWebEngineView(self)
        self.syn_signal_plot = QtWebEngineWidgets.QWebEngineView(self)
        self.filtered_syn_signal_plot = QtWebEngineWidgets.QWebEngineView(self)
        self.mag_response_plot = QtWebEngineWidgets.QWebEngineView(self)
        self.phase_response_plot = QtWebEngineWidgets.QWebEngineView(self)
        self.impulse_response_plot = QtWebEngineWidgets.QWebEngineView(self)
        self.step_response_plot = QtWebEngineWidgets.QWebEngineView(self)
        self.ui.SignalGraphLayout.addWidget(self.signal_plot)
        self.ui.SignalPropLayout.addWidget(self.signal_prop_plot)
        self.ui.SynSignalLayout.addWidget(self.syn_signal_plot)
        self.ui.FiltSynSignalLayout.addWidget(self.filtered_syn_signal_plot)
        self.ui.FreqResLayout.addWidget(self.mag_response_plot)
        self.ui.PhaseResLayout.addWidget(self.phase_response_plot)
        self.ui.ImpulseResLayout.addWidget(self.impulse_response_plot)
        self.ui.StepResLayout.addWidget(self.step_response_plot)
        self.ui.actionOpen_signal.triggered.connect(self.open_record)
        self.ui.actionOpen_Dataset.triggered.connect(self.open_dataset)
        self.ui.SignalFsSpinBox.valueChanged.connect(self.set_input_sig_freq)
        self.ui.actionPlot_PSd.triggered.connect(self.plot_psd)
        self.ui.actionPlot_Spectrogram.triggered.connect(self.plot_spectrogram)
        self.ui.actionPlot_CWT.triggered.connect(self.plot_cwt)
        self.ui.actionPlot_SSCWT.triggered.connect(self.plot_sscwt)
        self.ui.T_lowSpinBox.valueChanged.connect(self.syn_sig_plot)
        self.ui.T_highSpinBox.valueChanged.connect(self.syn_sig_plot)
        self.ui.NSpinBox.valueChanged.connect(self.syn_sig_plot)
        self.ui.FreqSpinBox.valueChanged.connect(self.syn_sig_plot)
        self.ui.Phi0SpinBox.valueChanged.connect(self.syn_sig_plot)
        self.ui.SNRSpinBox.valueChanged.connect(self.syn_sig_plot)
        self.ui.FcSpinBox.valueChanged.connect(self.syn_sig_plot)
        self.ui.AcSpinBox.valueChanged.connect(self.syn_sig_plot)
        self.ui.FaSpinBox.valueChanged.connect(self.syn_sig_plot)
        self.ui.KaSpinBox.valueChanged.connect(self.syn_sig_plot)
        self.ui.KfSpinBox.valueChanged.connect(self.syn_sig_plot)
        self.ui.OrderSpinBox.valueChanged.connect(self.build_plot_filter)
        self.ui.F_highSpinBox.valueChanged.connect(self.build_plot_filter)
        self.ui.F_lowSpinBox.valueChanged.connect(self.build_plot_filter)
        self.ui.FsSpinBox.valueChanged.connect(self.build_plot_filter)
        self.ui.RpSpinBox.valueChanged.connect(self.build_plot_filter)
        self.ui.RsSpinBox.valueChanged.connect(self.build_plot_filter)
        self.ui.SignalTypeComboBox.currentIndexChanged.connect(self.syn_sig_plot)
        self.ui.OutputComboBox.currentIndexChanged.connect(self.build_plot_filter)
        self.ui.BtypeComboBox.currentIndexChanged.connect(self.build_plot_filter)
        self.ui.FtypeComboBox.currentIndexChanged.connect(self.build_plot_filter)
        self.ui.WindowComboBox.currentIndexChanged.connect(self.build_plot_filter)
        self.ui.FilterMethodComboBox.currentIndexChanged.connect(self.build_plot_filter)
        self.ui.AMCheckBox.stateChanged.connect(self.syn_sig_plot)
        self.ui.FMCheckBox.stateChanged.connect(self.syn_sig_plot)
        self.ui.EndpointCheckBox.stateChanged.connect(self.syn_sig_plot)
        self.ui.S_impCheckBox.stateChanged.connect(self.syn_sig_plot)
        self.ui.AnalogCheckBox.stateChanged.connect(self.build_plot_filter)
        self.ui.ScaleCheckBox.stateChanged.connect(self.build_plot_filter)
        self.ui.ApplyFilterOriginalSignal.stateChanged.connect(self.apply_or_remove_filter)
        self.ui.FilesListWidget.currentItemChanged.connect(self.plot_selected_signal)
        self.ui.FiltersMainSplitter.setSizes([50, 200])
        self.ui.SynSignalMainSplitter.setSizes([50, 200])
        self.ui.Tab1MainSplitter.setSizes([30,300])

        self.in_sig_fs = self.ui.SignalFsSpinBox.value()
        self.root_dir_path = ''
        self.syn_signal = [0]
        self.filt = [0]
        self.t_low = 0
        self.t_high = 10
        self.f_low = 0.025
        self.f_high = 0.3
        self.filter_fs = 50
        self.syn_sig_freq = 10
        self.n = 5000
        self.phi0 = 0
        self.snr = None
        self.am = False
        self.fm = False
        self.s_imp = False
        self.endpoint = False
        self.kf = 0.25
        self.ka = 0.25
        self.fc = 50
        self.fa = 50
        self.ac = 1.0
        self.rs = None
        self.rp = None
        self.analog = False
        self.scale = True
        self.order = 2
        self.signaltype = 'sine'
        self.btype='bandpass'
        self.output = 'ba'
        self.filtermethod = 'iir'
        self.window = 'hamming'
        self.ftype = 'butter'
        self.record_chs = 3
        self.spectro_flag = 0
        self.cwt_flag = 0
        self.sscwt_flag = 0
        self.psd_flag = 0


    def open_record(self):
        files_name = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open only np or txt or CSV', os.getenv('HOME'), "csv(*.csv);; text(*.txt);; numpy(*.npy)")
        path = files_name[0]

        self.record = load_data.load_record(path,delimiter=',',dtype=None,ch_structure='v',target_ch_structure='v')
        self.record_chs = min(self.record.shape[0],self.record.shape[1])
        fig = viz.plot_signal(self.record,n_chs=self.record_chs,fs=self.in_sig_fs,gui=True)
        self.signal_plot.setHtml(fig.to_html(include_plotlyjs='cdn'))

    def open_dataset(self):
        dir_name = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Open a directory', os.getenv('HOME'), QFileDialog.ShowDirsOnly)
        for (root,dirs,files) in os.walk(dir_name,topdown=True):
            self.root_dir_path = root
            for file_name in files:
                listWidgetItem = QListWidgetItem(file_name)
                self.ui.FilesListWidget.addItem(listWidgetItem)
            self.record = load_data.load_record(os.path.join(self.root_dir_path,files[0]),delimiter=',',dtype=None,ch_structure='v',target_ch_structure='v')
            self.record_chs = min(self.record.shape[0],self.record.shape[1])
            self.ui.FilesListWidget.setCurrentItem(self.ui.FilesListWidget.item(0))
            self.apply_or_remove_filter()


    def replot_current_prop(self):
        if self.spectro_flag:
            self.plot_spectrogram()
        elif self.psd_flag:
            self.plot_psd()
        elif self.cwt_flag:
            self.plot_cwt()
        elif self.sscwt_flag:
            self.plot_sscwt()
    

    def set_input_sig_freq(self):
        time.sleep(1)
        self.in_sig_fs = self.ui.SignalFsSpinBox.value()
        if self.ui.ApplyFilterOriginalSignal.isChecked():
            fig = viz.plot_signal(self.filtered_original_signal,n_chs=self.record_chs,fs=self.in_sig_fs,gui=True)
        else:
            fig = viz.plot_signal(self.record,n_chs=self.record_chs,fs=self.in_sig_fs,gui=True)
        self.signal_plot.setHtml(fig.to_html(include_plotlyjs='cdn'))
        self.replot_current_prop()

    def plot_selected_signal(self):
        self.record = self.ui.FilesListWidget.currentItem().text()
        self.record = load_data.load_record(os.path.join(self.root_dir_path,self.ui.FilesListWidget.currentItem().text()),delimiter=',',dtype=None,ch_structure='v',target_ch_structure='v')
        self.record_chs = min(self.record.shape[0],self.record.shape[1])
        self.apply_or_remove_filter()
    
    def plot_psd(self):
        if self.ui.ApplyFilterOriginalSignal.isChecked():
            fig = viz.plot_psd(self.filtered_original_signal,n_chs=self.record_chs,fs=self.in_sig_fs,gui=True)
        else:
            fig = viz.plot_psd(self.record,n_chs=self.record_chs,fs=self.in_sig_fs,gui=True)
        self.signal_prop_plot.setHtml(fig.to_html(include_plotlyjs='cdn'))
        self.psd_flag = 1
        self.spectro_flag=0
        self.cwt_flag = 0
        self.sscwt_flag=0


    def plot_cwt(self):
        if self.ui.ApplyFilterOriginalSignal.isChecked():
            fig = viz.plot_sscwt_cwt(self.filtered_original_signal,n_chs=self.record_chs,fs=self.in_sig_fs,
        colormap = 'Viridis',type='cwt',plot_3d=False,gui=True)
        else:
             fig = viz.plot_sscwt_cwt(self.record,n_chs=self.record_chs,fs=self.in_sig_fs,
        colormap = 'Viridis',type='cwt',plot_3d=False,gui=True)

        fig_html = fig.to_html(include_plotlyjs='cdn')
        self.signal_prop_plot.setHtml(fig_html)
        self.signal_prop_plot.show()
        fig.show()
        self.psd_flag = 0
        self.spectro_flag=0
        self.cwt_flag = 1
        self.sscwt_flag=0

    def plot_sscwt(self):
        if self.ui.ApplyFilterOriginalSignal.isChecked():
            fig = viz.plot_sscwt_cwt(self.filtered_original_signal,n_chs=self.record_chs,fs=self.in_sig_fs,
        colormap = 'Viridis',type='sscwt',plot_3d=False,gui=True)
        else:
            fig = viz.plot_sscwt_cwt(self.record,n_chs=self.record_chs,fs=self.in_sig_fs,
                                    colormap = 'Viridis',type='sscwt',plot_3d=False,gui=True)
        fig_html = fig.to_html(include_plotlyjs='cdn')
        self.signal_prop_plot.setHtml(fig_html)
        self.signal_prop_plot.show()
        fig.show()
        self.psd_flag = 0
        self.spectro_flag=0
        self.cwt_flag = 0
        self.sscwt_flag=1

    def plot_spectrogram(self):
        if self.ui.ApplyFilterOriginalSignal.isChecked():
            fig = viz.plot_spectrogram(self.filtered_original_signal,n_chs=self.record_chs,fs=self.in_sig_fs,gui=True,
        colormap='Viridis')
        else:
            fig = viz.plot_spectrogram(self.record,n_chs=self.record_chs,fs=self.in_sig_fs,gui=True,
        colormap='Viridis')
        self.signal_prop_plot.setHtml(fig.to_html(include_plotlyjs='cdn'))
        self.psd_flag = 0
        self.spectro_flag=1
        self.cwt_flag = 0
        self.sscwt_flag=0

    def build_plot_filter(self):
        time.sleep(1)
        self.f_low = self.ui.F_lowSpinBox.value()
        self.f_high = self.ui.F_highSpinBox.value()
        self.filter_fs = self.ui.FsSpinBox.value()
        self.rs = self.ui.RsSpinBox.value()
        self.rp = self.ui.RpSpinBox.value()
        self.analog = self.ui.AnalogCheckBox.isChecked()
        self.scale = self.ui.ScaleCheckBox.isChecked()
        self.order = self.ui.OrderSpinBox.value()
        self.btype=self.ui.BtypeComboBox.currentText()
        self.output = self.ui.OutputComboBox.currentText()
        self.filtermethod = self.ui.FilterMethodComboBox.currentText()
        self.window = self.ui.WindowComboBox.currentText()
        self.ftype = self.ui.FtypeComboBox.currentText()
        b,a = 0,0
        w,h = 0,0
        z,p,k=0,0,0
        sos=None
        self.filt,w,h= filter.create_filter(fs= self.filter_fs,f_low=self.f_low,f_high=self.f_high,
                        order=self.order,method=self.filtermethod, rp=self.rp, rs=self.rs, 
                        btype=self.btype, analog=self.analog, ftype=self.ftype, 
                    window=self.window, scale=self.scale,output=self.output)
        if self.filtermethod == 'fir':
            b = self.filt
            a = 1
        else:
            if self.output == 'ba':
                b,a=self.filt
            elif self.output == 'sos':
                sos = self.filt
            elif self.output == 'zpk':
                z,p,k=self.filt
                sos = signal.zpk2sos(z, p, k, analog=self.analog)
        mag_res = viz.plot_mag_response(w,h,self.filter_fs,gui=True)
        phase_res = viz.plot_phase_response(w,h,self.filter_fs,gui=True)
        impulse_res = viz.plot_impulse_response(b,a,sos,gui=True)
        step_res = viz.plot_step_response(b,a,sos,gui=True)
        self.mag_response_plot.setHtml(mag_res.to_html(include_plotlyjs='cdn'))
        self.phase_response_plot.setHtml(phase_res.to_html(include_plotlyjs='cdn'))
        self.impulse_response_plot.setHtml(impulse_res.to_html(include_plotlyjs='cdn'))
        self.step_response_plot.setHtml(step_res.to_html(include_plotlyjs='cdn'))
        if np.array(self.syn_signal).any():
            self.syn_sig_filt_plot()


    def syn_sig_plot(self):
        time.sleep(1)
        self.t_low = self.ui.T_lowSpinBox.value()
        self.t_high = self.ui.T_highSpinBox.value()
        self.syn_sig_freq = self.ui.FreqSpinBox.value()
        self.n = self.ui.NSpinBox.value()
        self.phi0 = self.ui.Phi0SpinBox.value()
        self.am = self.ui.AMCheckBox.isChecked()
        self.fm = self.ui.FMCheckBox.isChecked()
        self.s_imp = self.ui.S_impCheckBox.isChecked()
        self.endpoint = self.ui.EndpointCheckBox.isChecked()
        self.kf = self.ui.KfSpinBox.value()
        self.ka = self.ui.KaSpinBox.value()
        self.fc = self.ui.FcSpinBox.value()
        self.fa = self.ui.FaSpinBox.value()
        self.ac = self.ui.AcSpinBox.value()
        self.signaltype = self.ui.SignalTypeComboBox.currentText()
        self.snr = self.ui.SNRSpinBox.value()

        self.syn_signal,self.syn_sig_time = generate_signal.gen_syn_signal(type=self.signaltype,n=self.n,t=(self.t_low,self.t_high),
                        f=self.syn_sig_freq,phi0=self.phi0,snr=self.snr,am=self.am,ac=self.ac, fa=self.fa,
                        ka=self.ka,fm=self.fm,fc=self.fc,kf=self.kf,s_imp=self.s_imp,endpoint=self.endpoint)
        syn_sig_fig = viz.plot_signal(self.syn_signal,t=self.syn_sig_time,n_chs=1,fs=(self.n/float(self.t_high)),gui=True)
        self.syn_signal_plot.setHtml(syn_sig_fig.to_html(include_plotlyjs='cdn'))
        if np.array(self.filt).any():
            self.syn_sig_filt_plot()



    def apply_filter_on_signal(self,sig,filt):
        if self.output == 'ba':
            b,a = filt
            response = signal.lfilter(b, a, sig)
        else:
            if self.output == 'zpk':
                z,p,k=filt
                sos = signal.zpk2sos(z, p, k, analog=self.analog)
            else:
                sos = filt
            response  = signal.sosfilt(sos,sig)
        return response


    def syn_sig_filt_plot(self):
        self.response = self.apply_filter_on_signal(self.syn_signal,self.filt)

        fig = viz.plot_signal(self.response,t=self.syn_sig_time,n_chs=1,fs=(self.n/float(self.t_high)),gui=True)
        self.filtered_syn_signal_plot.setHtml(fig.to_html(include_plotlyjs='cdn'))

    
    def apply_filter_on_original_signal(self):
        record_chs_data = separate_data_chs(self.record,self.record_chs)
        filtered_signal = []
        for i,ch in enumerate(record_chs_data):
            filt_ch = self.apply_filter_on_signal(ch,self.filt)
            filtered_signal.append(filt_ch)
        return np.array(filtered_signal)


    
    def apply_or_remove_filter(self):
        if self.ui.ApplyFilterOriginalSignal.isChecked():
            self.filtered_original_signal = self.apply_filter_on_original_signal()
            fig = viz.plot_signal(self.filtered_original_signal,n_chs=self.record_chs,fs=self.in_sig_fs,gui=True)
        else:
            fig = viz.plot_signal(self.record,n_chs=self.record_chs,fs=self.in_sig_fs,gui=True)
        self.signal_plot.setHtml(fig.to_html(include_plotlyjs='cdn'))
        self.replot_current_prop()
        


def window():
    app = QApplication(sys.argv)
    win = ApplicationWindow()
    win.show()
    sys.exit(app.exec_())


# main code
if __name__ == "__main__":
    window()